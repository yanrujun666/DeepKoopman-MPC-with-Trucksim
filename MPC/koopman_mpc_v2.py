"""
Koopman-MPC V2：新网络 + 新 MPC 架构（方案 A）
从 mpc_dk 拷贝/复用的实现，不依赖 mpc_dk 包，便于后续删除 mpc_dk 文件夹。

- 编码器：CustomEncoderUnscaledv2WithoutNorm（无 BN、无 sigmoid，含 skip）
- Koopman：Koopmanv1
- MPC：KoopmanMPC（12 维控制、硬约束、z 空间参考）
- 与 Simulink/Trucksim 对接：状态为相对起点坐标，直接编码；参考轨迹从 .mat 加载后编码为 z_ref。
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# --------------- 控制边界（与 mpc_dk 一致，numpy 便于接口） ---------------
CONTROL_MIN_NP = np.array([
    -1500, -1500, -1500, -1500, -1500, -1500,
    -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2
], dtype=np.float64)
CONTROL_MAX_NP = np.array([
    1500, 1500, 1500, 1500, 1500, 1500,
    np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2
], dtype=np.float64)

# --------------- Transformer 与编码器（从 mpc_dk/eval_sequence 拷贝） ---------------

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                   tgt_key_padding_mask=None, memory_key_padding_mask=None,
                   pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class BatchFourierPositionalEncoding(nn.Module):
    """无 sigmoid 的 Fourier PE，与 CustomEncoderUnscaledv2WithoutNorm 配套。"""
    def __init__(self, L=10, input_dim=2):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim * L
        self.register_buffer('freqs', 2.0 ** torch.arange(0, L, dtype=torch.float32))

    def forward(self, coords):
        coords_normalized = coords
        angle = coords_normalized.unsqueeze(-1) * self.freqs.view(1, 1, 1, 1, -1)
        sin_enc = torch.sin(2 * torch.pi * angle)
        cos_enc = torch.cos(2 * torch.pi * angle)
        pe = torch.stack([sin_enc, cos_enc], dim=-1)
        pe = pe.flatten(start_dim=-3, end_dim=-1)
        pe = torch.cat([coords, pe], dim=-1)
        return pe


class CustomEncoderUnscaledv2WithoutNorm(nn.Module):
    """新 ckpt 使用的编码器：无 BN、无 sigmoid、含 skip_fc。"""
    def __init__(self, state_dim, hidden_dim, layer_depth):
        super().__init__()
        self.in_embed = BatchFourierPositionalEncoding(L=8, input_dim=1)
        self.channel_fc = nn.Sequential(
            nn.Linear(16 + 1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU()
        )
        encoder_depth = 2
        decoder_depth = 2
        self.state_pe = nn.Embedding(state_dim, 128)
        encoder_layer = TransformerEncoderLayer(128, 8, 128, 0.0, 'gelu', False)
        self.state_encoder = TransformerEncoder(encoder_layer, encoder_depth)
        decoder_layer = TransformerDecoderLayer(128, 8, 128, 0.0, "gelu", False)
        decoder_norm = nn.LayerNorm(128)
        self.state_decoder = TransformerDecoder(decoder_layer, decoder_depth, decoder_norm, return_intermediate=False)
        self.out_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim - state_dim),
        )
        self.skip_fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim - state_dim),
        )

    def forward(self, x):
        B, S, C = x.shape
        x_embed = self.in_embed(x.unsqueeze(-1))
        x_embed = self.channel_fc(x_embed)
        B, S, N, C = x_embed.shape
        pos = self.state_pe.weight.repeat(B * S, 1, 1)
        x_embed_flat = x_embed.view(B * S, N, C)
        x_embed_flat = x_embed_flat.permute(1, 0, 2)
        pos_flat = pos.permute(1, 0, 2)
        x_embed_fused = self.state_encoder(x_embed_flat, pos=pos_flat)
        fused_embed = torch.zeros((1, 1, C), device=x_embed.device).repeat(B * S, 1, 1).permute(1, 0, 2)
        fused_embed = self.state_decoder(fused_embed, x_embed_fused, pos=pos_flat)[0, 0]
        fused_embed = fused_embed.view(B, S, C)
        skip_embed = self.skip_fc(x)
        return self.out_fc(fused_embed) + skip_embed


class Koopmanv1(nn.Module):
    """Koopman 动力学 z_{k+1} = z @ A + u @ B。"""
    def __init__(self, state_dim, hidden_dim, control_dim):
        super(Koopmanv1, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(1.0 / control_dim * torch.zeros(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters=1):
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ self.A + u_tmp @ self.B
            results.append(x_tmp)
        return torch.stack(results, dim=1)


# --------------- KoopmanMPC（从 mpc_dk/mpc_koopman_control 拷贝，用 numpy 边界） ---------------

class KoopmanMPC(nn.Module):
    """
    新 MPC：z_{k+1} = z A + u B，12 维控制，硬约束（扭矩一致、转向几何）。
    ref_traj 为 [H, hidden_dim] 的 z 空间参考。
    """
    def __init__(
        self,
        dkm: Koopmanv1,
        state_dim: int = 6,
        control_dim: int = 12,
        horizon: int = 30,
        control_min_np: Optional[np.ndarray] = None,
        control_max_np: Optional[np.ndarray] = None,
        q_weights=None,
        r_weights=None,
        i_weights=None,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.dkm = dkm
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.device = device

        cmin = control_min_np if control_min_np is not None else CONTROL_MIN_NP
        cmax = control_max_np if control_max_np is not None else CONTROL_MAX_NP
        steer_min = float(cmin[6])
        steer_max = float(cmax[6])
        self.steer_center_norm = float((0.0 - steer_min) / (steer_max - steer_min))
        self.steer_opposite_sum_norm = float((-2.0 * steer_min) / (steer_max - steer_min))

        self.u_min = torch.zeros(self.control_dim, device=device)
        self.u_max = torch.ones(self.control_dim, device=device)

        if q_weights is None:
            base = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
            tail = torch.ones(self.dkm.A.shape[0] - base.numel(), device=device)
            q_weights = torch.cat([base, tail], dim=0)
        if r_weights is None:
            # 对 u 偏离 0.5（中性）加小惩罚，避免解总贴边界
            r_weights = torch.ones(self.control_dim, device=device) * 0.01
        if i_weights is None:
            # 控制增量惩罚，加大可减轻步间控制抖动（原 0.1 -> 0.5）
            i_weights = torch.ones(self.control_dim, device=device) * 0.5

        self.register_buffer("q_weights", q_weights)
        self.register_buffer("r_weights", r_weights)
        self.register_buffer("i_weights", i_weights)

        with torch.no_grad():
            A = self.dkm.A.detach().cpu().numpy()
            B = self.dkm.B.detach().cpu().numpy()
        self.A_np = A
        self.B_np = B
        self.Q_np = np.diag(self.q_weights.detach().cpu().numpy())
        self.R_np = np.diag(self.r_weights.detach().cpu().numpy())
        self.I_np = np.diag(self.i_weights.detach().cpu().numpy())
        self.u_min_np = self.u_min.detach().cpu().numpy()
        self.u_max_np = self.u_max.detach().cpu().numpy()

    def _solve_mpc_qp_single(self, z0_np: np.ndarray, ref_traj_np: np.ndarray,
                             u_init_np: Optional[np.ndarray] = None) -> np.ndarray:
        """ref_traj_np: [H, hidden_dim]，与 z 同维。"""
        H = ref_traj_np.shape[0]
        n = z0_np.shape[0]
        m = self.control_dim
        A, B = self.A_np, self.B_np
        Q, R, I = self.Q_np, self.R_np, self.I_np

        z = cp.Variable((H + 1, n))
        u = cp.Variable((H, m))
        constraints = [z[0, :] == z0_np]
        for k in range(H):
            constraints.append(z[k + 1, :] == z[k, :] @ A + u[k, :] @ B)
            constraints.append(u[k, :] >= self.u_min_np)
            constraints.append(u[k, :] <= self.u_max_np)
            T_FL, T_FR, T_ML, T_MR, T_RL, T_RR = u[k, 0], u[k, 1], u[k, 2], u[k, 3], u[k, 4], u[k, 5]
            d_FL, d_FR, d_ML, d_MR, d_RL, d_RR = u[k, 6], u[k, 7], u[k, 8], u[k, 9], u[k, 10], u[k, 11]
            constraints += [T_FL == T_FR, T_FR == T_ML, T_ML == T_MR, T_MR == T_RL, T_RL == T_RR]
            constraints += [d_FL == d_FR, d_ML == d_MR, d_RL == d_RR]
            constraints += [d_ML == self.steer_center_norm, d_MR == self.steer_center_norm]
            constraints += [d_RL + d_FL == self.steer_opposite_sum_norm, d_RR + d_FR == self.steer_opposite_sum_norm]

        cost = 0
        if u_init_np is not None:
            u_neutral_np = u_init_np
        else:
            u_neutral_np = np.ones(m, dtype=np.float64) * 0.5
       
        for k in range(H):
            cost += cp.quad_form(z[k + 1, :] - ref_traj_np[k], Q)
            cost += cp.quad_form(u[k, :] - u_neutral_np, R)
        for k in range(1, H):
            cost += cp.quad_form(u[k, :] - u[k - 1, :], I)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        if u_init_np is not None:
            u.value = u_init_np
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            prob.solve(warm_start=True, verbose=True)
        # -------- DEBUG: QP 状态与解 --------
        if not hasattr(KoopmanMPC, "_debug_solve_count"):
            KoopmanMPC._debug_solve_count = 0
        if KoopmanMPC._debug_solve_count < 8:
            KoopmanMPC._debug_solve_count += 1
            status = getattr(prob, "status", "?")
            print(f"[MPC DEBUG] QP solve #{KoopmanMPC._debug_solve_count}: status={status}")
            if ref_traj_np.size > 0:
                print(f"  z0_np[:6]={z0_np[:6].round(6).tolist()}, ref_traj_np[0][:6]={ref_traj_np[0][:6].round(6).tolist()}")
            if z.value is not None and ref_traj_np.size > 0:
                err_0 = np.linalg.norm(z.value[1, :] - ref_traj_np[0])
                print(f"  tracking_err(z1-ref0)={err_0:.6f}")
            if u.value is not None:
                u0 = np.asarray(u.value[0, :])
                print(f"  u_opt[0] (norm): torque≈{u0[:6].mean():.4f}, steer_F≈{u0[6]:.4f} steer_M≈{u0[8]:.4f} steer_R≈{u0[10]:.4f}")
            else:
                print(f"  u.value is None (QP failed or infeasible)")
        if u.value is None:
            return np.ones((H, m), dtype=np.float32) * 0.5
        return np.asarray(u.value, dtype=np.float32)

    def optimize(self, z0: torch.Tensor, ref_traj_z: torch.Tensor,
                 u_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z0: [B, hidden_dim], ref_traj_z: [B, H, hidden_dim], u_init: [B, H, control_dim] 或 None
        返回 U_opt: [B, H, control_dim]
        """
        B, H, _ = ref_traj_z.shape
        assert B == 1, "当前仅支持 batch_size=1"
        z0_np = z0[0].detach().cpu().numpy()
        ref_np = ref_traj_z[0].detach().cpu().numpy()
        u_init_np = None
        if u_init is not None:
            u_init_np = u_init[0].detach().cpu().numpy()
        u_opt_np = self._solve_mpc_qp_single(z0_np, ref_np, u_init_np)
        return torch.from_numpy(u_opt_np).unsqueeze(0).to(self.device)


# --------------- 加载与接口（供 ddk_mpc_sfunction 使用） ---------------

def load_model(ckpt_path: str, device: Optional[torch.device] = None):
    """加载新 ckpt，返回 (encoder, dkm)。"""
    if device is None:
        device = torch.device("cpu")
    encoder = CustomEncoderUnscaledv2WithoutNorm(state_dim=6, hidden_dim=16, layer_depth=6)
    dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    dkm.load_state_dict(ckpt["koopman"])
    encoder = encoder.to(device)
    dkm = dkm.to(device)
    encoder.eval()
    dkm.eval()
    return encoder, dkm


def encode_state(encoder: CustomEncoderUnscaledv2WithoutNorm, x_6: np.ndarray,
                device: torch.device) -> np.ndarray:
    """单状态编码：x_6 (6,) 或 (1,6) -> z (16,)。"""
    x = np.asarray(x_6, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, 1, 6)
    elif x.ndim == 2:
        x = x.reshape(1, -1, 6)
    t = torch.from_numpy(x).to(device)
    with torch.no_grad():
        lift = encoder(t)
    lift_np = lift.cpu().numpy()
    state_np = t.cpu().numpy()
    z = np.concatenate([state_np.reshape(-1, 6), lift_np.reshape(-1, 10)], axis=1)
    return z.flatten()


def encode_ref_trajectory(encoder: CustomEncoderUnscaledv2WithoutNorm, ref_6: np.ndarray,
                         device: torch.device) -> np.ndarray:
    """参考轨迹编码：ref_6 (Np, 6) -> z_ref (Np, 16)。"""
    ref_6 = np.asarray(ref_6, dtype=np.float32)
    if ref_6.ndim == 1:
        ref_6 = ref_6.reshape(1, 6)
    Np = ref_6.shape[0]
    t = torch.from_numpy(ref_6).unsqueeze(0).to(device)
    with torch.no_grad():
        lift = encoder(t)
    lift_np = lift.cpu().numpy().reshape(Np, 10)
    z_ref = np.concatenate([ref_6, lift_np], axis=1)
    return z_ref.astype(np.float32)


def convert_to_control_output(u_normalized: np.ndarray,
                              control_min_np: Optional[np.ndarray] = None,
                              control_max_np: Optional[np.ndarray] = None) -> np.ndarray:
    """
    归一化控制 [0,1]^12 -> Trucksim 格式 [steer_LF..RR(deg), torque_LF..RR(N·m)]。
    输入 u 顺序：前 6 转矩，后 6 转向角(rad)。
    """
    cmin = control_min_np if control_min_np is not None else CONTROL_MIN_NP
    cmax = control_max_np if control_max_np is not None else CONTROL_MAX_NP
    u = np.asarray(u_normalized, dtype=np.float64).ravel()
    if len(u) != 12:
        raise ValueError(f"期望 12 维控制，得到 {len(u)}")
    u_phys = u * (cmax - cmin) + cmin
    torques = u_phys[:6]
    steer_rad = u_phys[6:12]
    steer_deg = steer_rad * 180.0 / np.pi
    return np.array([
        steer_deg[0], steer_deg[1], steer_deg[2], steer_deg[3], steer_deg[4], steer_deg[5],
        torques[0], torques[1], torques[2], torques[3], torques[4], torques[5]
    ], dtype=np.float64)
