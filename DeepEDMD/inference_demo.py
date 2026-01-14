import os
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import copy
from typing import Optional, List
import torch.nn.functional as F
from torch import Tensor

from torch.utils.data import ConcatDataset

from src.loader import VehicleDynamicDataset
from src.Koopman import Koopmanv1


class BatchFourierPositionalEncodingv2(nn.Module):
    def __init__(self, L=10, input_dim=2):
        """
        :param L: 频率级数，控制输出维度 C = 2 * input_dim * L
        :param input_dim: 输入坐标维度（默认为2，即x和y）
        """
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim * L
        
        # 预计算频率 2^k (k=0, 1, ..., L-1)
        self.register_buffer('freqs', 2.0 ** torch.arange(0, L, dtype=torch.float32))

    def forward(self, coords):
        """
        :param coords: 输入坐标，形状为 (Batch_size, ins_num, pts_num, input_dim)
        :return: 编码后的特征，形状为 (Batch_size, ins_num, pts_num, output_dim)
        """       
        # 计算所有频率的正弦和余弦
        # angle: (Batch_size, ins_num, pts_num, input_dim, L)
        coords_normalized = coords.sigmoid()
        angle = coords_normalized.unsqueeze(-1) * self.freqs.view(1, 1, 1, 1, -1)
        sin_enc = torch.sin(2 * torch.pi * angle)  # (..., input_dim, L)
        cos_enc = torch.cos(2 * torch.pi * angle)  # (..., input_dim, L)
        
        # 将正弦和余弦交错排列
        pe = torch.stack([sin_enc, cos_enc], dim=-1)  # (..., input_dim, L, 2)
        pe = pe.flatten(start_dim=-3, end_dim=-1)     # (..., input_dim * L * 2)
        pe = torch.cat([coords, coords_normalized, pe], dim=-1)
        
        return pe
    

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
        # Implementation of Feedforward model
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

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
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

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
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

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
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

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CustomEncoderUnscaledv2(nn.Module):

    def __init__(self, state_dim, hidden_dim, layer_depth):
        super().__init__()

        self.in_bn = nn.BatchNorm1d(state_dim)

        self.in_embed = BatchFourierPositionalEncodingv2(L=8, input_dim=1)

        self.channel_fc = nn.Sequential(
            nn.Linear(16+2, 64),
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

        self.out_fc =nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim-state_dim),
        )
        
    def forward(self, x):

        B, S, C = x.shape
        x = x.view(B*S, -1)
        x_norm = self.in_bn(x)
        x = x_norm.view(B, S, C)
        
        x_embed = self.in_embed(x.unsqueeze(-1)) # B S state_dim C

        x_embed = self.channel_fc(x_embed)

        B, S, N, C = x_embed.shape
        pos = self.state_pe.weight.repeat(B*S, 1, 1)
        x_embed_flat = x_embed.view(B*S, N, C)
        x_embed_flat = x_embed_flat.permute(1, 0, 2)
        pos_flat = pos.permute(1, 0, 2)
        x_embed_fused = self.state_encoder(x_embed_flat, pos=pos_flat)

        fused_embed = torch.zeros((1, 1, C), device=x_embed.device).repeat(B*S, 1, 1).permute(1, 0, 2)
        fused_embed = self.state_decoder(fused_embed, x_embed_fused, pos=pos_flat)[0, 0]
        fused_embed = fused_embed.view(B, S, C)

        return self.out_fc(fused_embed)


class Koopmanv1(nn.Module):
    """
    Deep Neural Networks With Koopman Operators for Modeling and Control of Autonomous Vehicles, T-IV 2023
    """

    def __init__(self, state_dim, hidden_dim, control_dim):
        super(Koopmanv1, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(1 / control_dim * torch.zeros(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)

#######################

DEVICE = torch.device('cuda:0')
CONTROL_MIN = torch.tensor([-1500, -1500, -1500, -1500, -1500, -1500, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]).to(DEVICE)
CONTROL_MAX = torch.tensor([1500, 1500, 1500, 1500, 1500, 1500, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]).to(DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有GPU

set_seed(42)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def viz(est_states, figname):
    states_npy = est_states[0].cpu().numpy()
            
    timestamps = np.arange(len(states_npy)) * 0.01
            
    # visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    fig.suptitle('visualization of DeepKoopman', fontsize=16, fontweight='bold')
            
    axes_flat = axes.flatten()

    axes_flat[0].plot(timestamps, states_npy[:, 0], 'b-', linewidth=2, label='gt')
    axes_flat[0].set_title(f'X', fontsize=12, fontweight='bold')
    axes_flat[0].set_xlabel('timestamp')
    axes_flat[0].set_ylabel('x')
    axes_flat[0].grid(True, alpha=0.3)
    axes_flat[0].legend(loc='best')

    axes_flat[1].plot(timestamps, states_npy[:, 1], 'b-', linewidth=2, label='gt')
    axes_flat[1].set_title(f'Y', fontsize=12, fontweight='bold')
    axes_flat[1].set_xlabel('timestamp')
    axes_flat[1].set_ylabel('y')
    axes_flat[1].grid(True, alpha=0.3)
    axes_flat[1].legend(loc='best')

    axes_flat[2].plot(timestamps, states_npy[:, 2], 'b-', linewidth=2, label='gt')
    axes_flat[2].set_title(f'Yaw', fontsize=12, fontweight='bold')
    axes_flat[2].set_xlabel('timestamp')
    axes_flat[2].set_ylabel('yaw')
    axes_flat[2].grid(True, alpha=0.3)
    axes_flat[2].legend(loc='best')

    axes_flat[3].plot(states_npy[:, 0], states_npy[:, 1], 'b-', linewidth=2, label='gt')
    axes_flat[3].set_title(f'trajectory', fontsize=12, fontweight='bold')
    axes_flat[3].set_xlabel('x')
    axes_flat[3].set_ylabel('y')
    axes_flat[3].grid(True, alpha=0.3)
    axes_flat[3].legend(loc='best')

    axes_flat[4].plot(timestamps, states_npy[:, 3], 'b-', linewidth=2, label='gt')
    axes_flat[4].set_title(f'Vx', fontsize=12, fontweight='bold')
    axes_flat[4].set_xlabel('timestamp')
    axes_flat[4].set_ylabel('vx')
    axes_flat[4].grid(True, alpha=0.3)
    axes_flat[4].legend(loc='best')

    axes_flat[5].plot(timestamps, states_npy[:, 4], 'b-', linewidth=2, label='gt')
    axes_flat[5].set_title(f'Vy', fontsize=12, fontweight='bold')
    axes_flat[5].set_xlabel('timestamp')
    axes_flat[5].set_ylabel('vy')
    axes_flat[5].grid(True, alpha=0.3)
    axes_flat[5].legend(loc='best')

    axes_flat[6].plot(timestamps, states_npy[:, 5], 'b-', linewidth=2, label='gt')
    axes_flat[6].set_title(f'Wz', fontsize=12, fontweight='bold')
    axes_flat[6].set_xlabel('timestamp')
    axes_flat[6].set_ylabel('wz')
    axes_flat[6].grid(True, alpha=0.3)
    axes_flat[6].legend(loc='best')

    plt.tight_layout()

    plt.savefig(figname)
    plt.close()

if __name__ == '__main__':
    # config model
    def build_model(ckpt_path):
        encoder = CustomEncoderUnscaledv2(state_dim=6, hidden_dim=16, layer_depth=6)
        dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)

        print('Params of encoder:', count_parameters(encoder) / 1e6, 'M')
        print('Params of dkm:', count_parameters(dkm) / 1e6, 'M')

        ckpt = torch.load(ckpt_path)
        encoder.load_state_dict(ckpt['encoder'])
        dkm.load_state_dict(ckpt['koopman'])

        encoder = encoder.to(DEVICE).eval()
        dkm = dkm.to(DEVICE).eval()
        return encoder, dkm

    encoder, dkm = build_model(ckpt_path='ckpt/DeepEDMD-Transv2-hd16-multiset-100e.pth')

    ITERS = 100
    # t0时刻初始状态 x, y, yaw, vx, vy, wz
    states_t0 = torch.tensor([[0, 0, 0, 10, 0, 0]]).unsqueeze(0).to(DEVICE).float() # shape Bx1x6
    # t0~tN-1时刻控制输入 6 wheel T_motor, 6 wheel delta
    controls_t0_tN = torch.tensor([100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(0).repeat(1, ITERS, 1).to(DEVICE).float() # shape BxNx12
    print(states_t0.shape, controls_t0_tN.shape)
    with torch.no_grad():
        controls_norm_t0_tN = (controls_t0_tN - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)
        state_embeds_t0 = encoder(states_t0)
        state_embeds_full_t0= torch.cat([states_t0, state_embeds_t0], dim=-1)
        result_embeds_full_t1_tN1 = dkm(state_embeds_full_t0, controls_norm_t0_tN, iters=ITERS)
        # t1~tN时刻预测状态
        est_states_t1_tN1 = result_embeds_full_t1_tN1[..., :6]
        print(est_states_t1_tN1.shape)
        # print(est_states_t1_tN1)

    viz(est_states_t1_tN1, 'tmp.png')