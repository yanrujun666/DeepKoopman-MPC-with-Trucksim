"""
DeepEDMD-MPC控制器Python实现
实现DeepEDMD模型加载、编码、归一化等功能，以及MPC控制器
支持PyTorch Transformer编码器（Trucksim版本）
仅支持.pth格式模型文件
"""

import os
import sys

# 解决OpenMP冲突问题（必须在导入numpy之前设置）
# MATLAB和Python的科学计算库都使用了OpenMP，会导致冲突
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from typing import Tuple, Optional
import copy

# PyTorch相关导入（可选，如果使用pth格式模型）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[警告] PyTorch未安装，无法使用pth格式模型。请安装: pip install torch")

# ========== Transformer编码器相关类（从inference_demo.py复制） ==========

if TORCH_AVAILABLE:
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
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


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

            self.out_fc = nn.Sequential(
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


class DeepEDMD:
    """
    DeepEDMD模型类
    实现编码器、归一化、参考轨迹处理等功能
    仅支持PyTorch模型（.pth格式）
    """
    
    def __init__(self, param_path: str, device: str = 'cpu'):
        """
        初始化DeepEDMD模型
        
        Args:
            param_path: 参数文件路径（必须是.pth格式）
            device: PyTorch设备（'cpu'或'cuda'）
        
        Raises:
            ValueError: 如果文件不是.pth格式
            ImportError: 如果PyTorch未安装
        """
        if not param_path.endswith('.pth'):
            raise ValueError(f"仅支持.pth格式模型文件，当前文件: {param_path}")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法加载.pth格式模型。请安装: pip install torch")
        
        self.param_path = param_path
        self.device = device
        self._load_pytorch_model()
    
    def _load_pytorch_model(self):
        """加载PyTorch模型（.pth格式）"""
        print(f"[DeepEDMD] 加载PyTorch模型: {self.param_path}")
        
        # 模型参数（与inference_demo.py保持一致）
        self.state_dim = 6
        self.hidden_dim = 16
        self.control_dim = 12
        self.lift_dim = self.hidden_dim - self.state_dim  # 10
        self.space_dim = self.hidden_dim  # 16 = state_dim + lift_dim
        
        # 为了兼容MPCController中的代码，保留s_dim和u_dim作为别名（向后兼容）
        self.s_dim = self.state_dim  # 别名：s_dim = state_dim
        self.u_dim = self.control_dim  # 别名：u_dim = control_dim
        
        # 控制输入归一化边界（与inference_demo.py保持一致）
        # CONTROL_MIN: 6个车轮转矩 + 6个车轮转向角
        # 转矩范围: [-1500, 1500] N·m
        # 转向角范围: [-π/2, π/2] rad
        self.control_min = np.array([-1500, -1500, -1500, -1500, -1500, -1500, 
                                     -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        self.control_max = np.array([1500, 1500, 1500, 1500, 1500, 1500,
                                     np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
        
        # 加载checkpoint
        ckpt = torch.load(self.param_path, map_location=self.device)
        
        # 创建模型
        self.encoder_model = CustomEncoderUnscaledv2(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            layer_depth=6
        )
        self.koopman = Koopmanv1(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            control_dim=self.control_dim
        )
        
        # 加载权重
        self.encoder_model.load_state_dict(ckpt['encoder'])
        self.koopman.load_state_dict(ckpt['koopman'])

        # 设置为评估模式
        self.encoder_model.eval()
        self.koopman.eval()

        # 移动到指定设备
        self.encoder_model = self.encoder_model.to(self.device)
        self.koopman = self.koopman.to(self.device)
        
        # 提取Koopman矩阵A和B（转换为numpy格式，用于MPC）
        with torch.no_grad():
            self.A = self.koopman.A.cpu().numpy().T  # 转置以匹配MPC格式
            self.B = self.koopman.B.cpu().numpy().T  # 转置以匹配MPC格式
        
        # ========== Debug信息打印 ==========
        self._print_debug_info()
    
    def _print_debug_info(self):
        """打印Debug信息：A、B矩阵和Encoder网络参数"""
        print("\n" + "="*80)
        print("[DeepEDMD Debug] 模型参数信息")
        print("="*80)
        
        # 1. 打印网络结构参数
        print("\n[1] Encoder网络结构参数:")
        print(f"  state_dim (状态维度): {self.state_dim}")
        print(f"  hidden_dim (隐藏维度): {self.hidden_dim}")
        print(f"  control_dim (控制维度): {self.control_dim}")
        print(f"  lift_dim (提升维度): {self.lift_dim}")
        print(f"  space_dim (空间维度): {self.space_dim}")
        print(f"  设备 (device): {self.device}")
        
        # Encoder网络详细结构
        print("\n[2] Encoder网络详细结构:")
        print(f"  BatchNorm1d: input_features={self.state_dim}")
        print(f"  BatchFourierPositionalEncoding: L=8, input_dim=1, output_dim={2*1*8+2}={16+2}")
        print(f"  Channel FC: {16+2} -> 64 -> 128")
        print(f"  State PE: Embedding({self.state_dim}, 128)")
        print(f"  Transformer Encoder: d_model=128, nhead=8, num_layers=2")
        print(f"  Transformer Decoder: d_model=128, nhead=8, num_layers=2")
        print(f"  Output FC: 128 -> 64 -> {self.hidden_dim - self.state_dim}")
        
        # 控制输入归一化边界
        print("\n[3] 控制输入归一化边界:")
        print(f"  CONTROL_MIN (前6个转矩, 后6个转向角):")
        print(f"    转矩范围: [{self.control_min[0]:.2f}, {self.control_max[0]:.2f}] N·m")
        print(f"    转向角范围: [{self.control_min[6]:.4f}, {self.control_max[6]:.4f}] rad")
        print(f"    转向角范围: [{self.control_min[6]*180/np.pi:.2f}, {self.control_max[6]*180/np.pi:.2f}] deg")
        
        # 2. 打印A矩阵信息
        print("\n[4] Koopman矩阵A:")
        print(f"  形状: {self.A.shape}")
        print(f"  完整矩阵:")
        np.set_printoptions(precision=6, suppress=True, linewidth=120)
        print(self.A)
        np.set_printoptions()  # 恢复默认设置
        
        # A矩阵统计信息
        print(f"\n  A矩阵统计信息:")
        print(f"    均值: {np.mean(self.A):.8f}")
        print(f"    标准差: {np.std(self.A):.8f}")
        print(f"    最大值: {np.max(self.A):.8f} (位置: {np.unravel_index(np.argmax(self.A), self.A.shape)})")
        print(f"    最小值: {np.min(self.A):.8f} (位置: {np.unravel_index(np.argmin(self.A), self.A.shape)})")
        print(f"    对角线元素均值: {np.mean(np.diag(self.A)):.8f}")
        print(f"    对角线元素标准差: {np.std(np.diag(self.A)):.8f}")
        
        # A矩阵特征值
        try:
            eigvals_A = np.linalg.eigvals(self.A)
            print(f"    特征值范围: [{np.min(eigvals_A.real):.8f}, {np.max(eigvals_A.real):.8f}] (实部)")
            if np.any(eigvals_A.imag != 0):
                print(f"    特征值虚部范围: [{np.min(eigvals_A.imag):.8f}, {np.max(eigvals_A.imag):.8f}]")
            print(f"    最大特征值模长: {np.max(np.abs(eigvals_A)):.8f}")
            print(f"    条件数: {np.linalg.cond(self.A):.8f}")
            
            # 检查稳定性（特征值是否在单位圆内）
            max_eig_abs = np.max(np.abs(eigvals_A))
            if max_eig_abs < 1.0:
                print(f"    稳定性: 稳定 (最大特征值模长 < 1.0)")
            else:
                print(f"    稳定性: 可能不稳定 (最大特征值模长 = {max_eig_abs:.8f} >= 1.0)")
        except Exception as e:
            print(f"    特征值计算失败: {e}")
        
        # 3. 打印B矩阵信息
        print("\n[5] Koopman矩阵B:")
        print(f"  形状: {self.B.shape}")
        print(f"  完整矩阵:")
        np.set_printoptions(precision=6, suppress=True, linewidth=120)
        print(self.B)
        np.set_printoptions()  # 恢复默认设置
        
        # B矩阵统计信息
        print(f"\n  B矩阵统计信息:")
        print(f"    均值: {np.mean(self.B):.8f}")
        print(f"    标准差: {np.std(self.B):.8f}")
        print(f"    最大值: {np.max(self.B):.8f} (位置: {np.unravel_index(np.argmax(self.B), self.B.shape)})")
        print(f"    最小值: {np.min(self.B):.8f} (位置: {np.unravel_index(np.argmin(self.B), self.B.shape)})")
        
        # B矩阵每列的统计信息（对应每个控制输入）
        print(f"  各控制输入对应的B矩阵列统计:")
        for i in range(self.B.shape[1]):
            col = self.B[:, i]
            print(f"    控制输入 {i}: 均值={np.mean(col):.8f}, 标准差={np.std(col):.8f}, "
                  f"最大={np.max(col):.8f}, 最小={np.min(col):.8f}")
        
        # 4. 打印Encoder网络权重统计信息
        print("\n[6] Encoder网络权重统计信息:")
        total_params = 0
        trainable_params = 0
        
        for name, param in self.encoder_model.named_parameters():
            param_np = param.detach().cpu().numpy()
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
                
            print(f"  {name}:")
            print(f"    形状: {param.shape}")
            print(f"    参数数量: {param_count:,}")
            print(f"    可训练: {param.requires_grad}")
            print(f"    均值: {np.mean(param_np):.8f}")
            print(f"    标准差: {np.std(param_np):.8f}")
            print(f"    最大值: {np.max(param_np):.8f}")
            print(f"    最小值: {np.min(param_np):.8f}")
            print(f"    L2范数: {np.linalg.norm(param_np):.8f}")
            
            # 对于BatchNorm，打印额外的统计信息
            if 'bn' in name.lower() or 'norm' in name.lower():
                if 'weight' in name:
                    print(f"    类型: 缩放参数 (scale)")
                elif 'bias' in name:
                    print(f"    类型: 偏移参数 (shift)")
        
        print(f"\n  Encoder总参数数量: {total_params:,}")
        print(f"  可训练参数数量: {trainable_params:,}")
        
        # 5. 打印Koopman网络权重统计信息
        print("\n[7] Koopman网络权重统计信息:")
        koopman_total_params = 0
        koopman_trainable_params = 0
        
        for name, param in self.koopman.named_parameters():
            param_np = param.detach().cpu().numpy()
            param_count = param.numel()
            koopman_total_params += param_count
            if param.requires_grad:
                koopman_trainable_params += param_count
                
            print(f"  {name}:")
            print(f"    形状: {param.shape}")
            print(f"    参数数量: {param_count:,}")
            print(f"    可训练: {param.requires_grad}")
            print(f"    均值: {np.mean(param_np):.8f}")
            print(f"    标准差: {np.std(param_np):.8f}")
            print(f"    最大值: {np.max(param_np):.8f}")
            print(f"    最小值: {np.min(param_np):.8f}")
            print(f"    L2范数: {np.linalg.norm(param_np):.8f}")
            
            # A和B矩阵的特殊信息
            if name == 'A':
                print(f"    类型: Koopman算子A矩阵")
                print(f"    对角线元素: {np.diag(param_np)}")
            elif name == 'B':
                print(f"    类型: Koopman算子B矩阵")
        
        print(f"\n  Koopman总参数数量: {koopman_total_params:,}")
        print(f"  可训练参数数量: {koopman_trainable_params:,}")
        
        # 6. 打印模型总览
        print("\n[8] 模型总览:")
        print(f"  总参数数量: {total_params + koopman_total_params:,}")
        print(f"  总可训练参数数量: {trainable_params + koopman_trainable_params:,}")
        print(f"  模型文件路径: {self.param_path}")
        print(f"  模型状态: 评估模式 (eval)")
        
        print("\n" + "="*80)
        print("[DeepEDMD Debug] 信息打印完成")
        print("="*80 + "\n")
    
    def normalization(self, x: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
        """
        [已废弃] 坐标变换：将全局坐标转换为局部坐标
        
        注意：此方法已废弃。当前模型在全局坐标系中训练，不再需要坐标变换。
        请直接使用全局状态调用 encoder() 方法。
        
        Args:
            x: 全局状态 [X, Y, Yaw, vx, vy, yaw_rate] (6维)
            ref_pos: 参考点位置 [X, Y, Yaw] (3维)
        
        Returns:
            局部状态 [x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local] (6维)
            注意：不进行归一化，编码器内部有BatchNorm
        
        Raises:
            ValueError: 如果输入维度不正确
        """
        # 输入验证
        x = np.asarray(x)
        ref_pos = np.asarray(ref_pos)
        
        if len(x) < 6:
            raise ValueError(f"输入状态维度错误: 期望至少6维，实际{len(x)}维")
        if len(ref_pos) < 3:
            raise ValueError(f"参考位置维度错误: 期望至少3维，实际{len(ref_pos)}维")
        
        # 提取位置和速度
        global_pos = x[:3]  # [X, Y, Yaw]
        global_vel = x[3:6]  # [vx, vy, yaw_rate]
        
        # 计算相对位置
        dx = global_pos[0] - ref_pos[0]
        dy = global_pos[1] - ref_pos[1]
        dyaw = global_pos[2] - ref_pos[2]
        
        # 旋转矩阵（将全局坐标转换为局部坐标）
        cos_yaw = np.cos(ref_pos[2])
        sin_yaw = np.sin(ref_pos[2])
        
        # 局部位置
        x_local = dx * cos_yaw + dy * sin_yaw
        y_local = -dx * sin_yaw + dy * cos_yaw
        yaw_local = dyaw
        
        # 局部速度（直接使用全局速度，不进行坐标变换）
        vx_local = global_vel[0]
        vy_local = global_vel[1]
        yaw_rate_local = global_vel[2]
        
        # 构建局部状态 [x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local] (6维)
        local_state = np.array([x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local])
        
        return local_state
    
    def normalize_control(self, u: np.ndarray) -> np.ndarray:
        """
        控制输入归一化（使用CONTROL_MIN/MAX，与inference_demo.py保持一致）
        
        Args:
            u: 原始控制输入 [12维]
               前6维：6个车轮转矩 (N·m)，范围 [-1500, 1500]
               后6维：6个车轮转向角 (rad)，范围 [-π/2, π/2]
        
        Returns:
            归一化后的控制输入 [12维]，范围 [0, 1]（线性归一化）
        
        Raises:
            ValueError: 如果输入维度不正确或归一化范围无效
        """
        u = np.asarray(u)
        if len(u) != self.control_dim:
            raise ValueError(f"控制输入维度错误: 期望{self.control_dim}维，实际{len(u)}维")
        
        # 检查归一化范围的有效性
        control_range = self.control_max - self.control_min
        if np.any(control_range <= 0):
            raise ValueError(f"控制输入归一化范围无效: control_max - control_min <= 0")
        if np.any(control_range < 1e-10):
            raise ValueError(f"控制输入归一化范围过小，可能导致数值不稳定")
        
        # 使用CONTROL_MIN/MAX归一化
        u_norm = (u - self.control_min) / control_range
        return u_norm
    
    def denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        """
        控制输入反归一化
        
        Args:
            u_norm: 归一化后的控制输入 [12维]
        
        Returns:
            原始控制输入 [12维]
        
        Raises:
            ValueError: 如果输入维度不正确或归一化范围无效
        """
        u_norm = np.asarray(u_norm)
        if len(u_norm) != self.control_dim:
            raise ValueError(f"归一化控制输入维度错误: 期望{self.control_dim}维，实际{len(u_norm)}维")
        
        # 检查归一化范围的有效性
        control_range = self.control_max - self.control_min
        if np.any(control_range <= 0):
            raise ValueError(f"控制输入归一化范围无效: control_max - control_min <= 0")
        if np.any(control_range < 1e-10):
            raise ValueError(f"控制输入归一化范围过小，可能导致数值不稳定")
        
        # 使用CONTROL_MIN/MAX反归一化
        u = u_norm * control_range + self.control_min
        return u
    
    def get_reference(self, ref_traj: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
        """
        [已废弃] 处理参考轨迹：转换到局部坐标系
        
        注意：此方法已废弃。当前模型在全局坐标系中训练，不再需要坐标变换。
        请直接使用全局参考轨迹，无需调用此方法。
        
        Args:
            ref_traj: 参考轨迹 [N, 6] 每行为 [X, Y, Yaw, vx, vy, yaw_rate]（原始值，未归一化）
            ref_pos: 参考点位置 [X, Y, Yaw]
        
        Returns:
            局部状态 [N, 6] 每行为局部状态（未归一化，编码器内部有BatchNorm）
        
        Raises:
            ValueError: 如果输入维度不正确
        """
        ref_traj = np.asarray(ref_traj)
        if ref_traj.ndim != 2 or ref_traj.shape[1] != self.state_dim:
            raise ValueError(f"参考轨迹维度错误: 期望(N, {self.state_dim})，实际{ref_traj.shape}")
        
        # 使用列表推导式优化效率，直接对每个参考点进行坐标变换（不归一化）
        ref_normalized = np.array([self.normalization(ref_traj[i, :], ref_pos) 
                                   for i in range(ref_traj.shape[0])])
        return ref_normalized
    
    def encoder(self, x: np.ndarray) -> np.ndarray:
        """
        编码器：将状态编码到提升空间
        
        Args:
            x: 状态 [x, y, yaw, vx, vy, yaw_rate] (6维) 或 [1, 6]
                输入原始状态（编码器内部有BatchNorm归一化）
        
        Returns:
            提升状态 [space_dim] = [state_dim + lift_dim] = [6 + 10] = 16维
        
        Raises:
            ValueError: 如果输入维度不正确
        """
        x = np.asarray(x)
        
        # 输入验证
        if x.ndim == 1:
            if len(x) != self.state_dim:
                raise ValueError(f"输入状态维度错误: 期望{self.state_dim}维，实际{len(x)}维")
            x = x.reshape(1, -1)
        elif x.ndim == 2:
            if x.shape[1] != self.state_dim:
                raise ValueError(f"输入状态维度错误: 期望(N, {self.state_dim})，实际{x.shape}")
        else:
            raise ValueError(f"输入状态维度错误: 期望1维或2维数组，实际{x.ndim}维")
        
        # 转换为torch tensor
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        # 添加序列维度: (B, S, C) = (1, 1, 6)
        x_tensor = x_tensor.unsqueeze(1)  # (B, 1, 6)
        
        # 编码器前向传播
        with torch.no_grad():
            # 编码器输出提升状态 (B, S, lift_dim)
            lift_state = self.encoder_model(x_tensor)  # (1, 1, 10)
            
            # 拼接原始状态和提升状态
            x_state = x_tensor  # (1, 1, 6)
            lifted_state_full = torch.cat([x_state, lift_state], dim=-1)  # (1, 1, 16)
            
            # 转换为numpy并返回
            result = lifted_state_full.cpu().numpy().flatten()  # (16,)
            return result


class MPCController:
    """
    MPC控制器类
    实现MPC优化求解
    """
    
    def __init__(self, deepedmd: DeepEDMD, Np: int = 30, Nc: int = 30, 
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                 delta_umax: Optional[np.ndarray] = None,
                 model_dt: float = 0.01, mpc_dt: Optional[float] = None,
                 sample_interval: Optional[int] = None):
        """
        初始化MPC控制器
        
        Args:
            deepedmd: DeepEDMD模型实例
            Np: 预测时域
            Nc: 控制时域
            Q: 状态权重矩阵 (state_dim x state_dim)
            R: 控制权重矩阵 (control_dim x control_dim)
            delta_umax: 控制增量约束 [control_dim]
            model_dt: DeepEDMD模型的时间步长（秒），默认0.01s（10ms）
            mpc_dt: MPC的采样时间（秒）。如果为None，则根据sample_interval计算
            sample_interval: 参考轨迹采样间隔（点数）。如果提供，mpc_dt = sample_interval * model_dt
        """
        self.deepedmd = deepedmd
        self.Np = int(Np)  # 确保是整数（预测时域）
        self.Nc = int(Nc)  # 确保是整数（控制时域）
        self.Nx = int(deepedmd.space_dim)  # 提升空间维度，确保是整数
        self.Nu = int(deepedmd.u_dim)      # 控制输入维度，确保是整数（使用u_dim别名）
        
        # 时间步长参数
        self.model_dt = float(model_dt)  # 模型时间步长（默认0.01s）
        
        # 计算MPC采样时间
        if mpc_dt is not None:
            self.mpc_dt = float(mpc_dt)
        elif sample_interval is not None:
            self.mpc_dt = float(sample_interval) * self.model_dt
        else:
            # 默认值：假设sample_interval=5
            self.mpc_dt = 5.0 * self.model_dt
        
        # 计算时间步长比例（必须是整数）
        self.time_step_ratio = int(round(self.mpc_dt / self.model_dt))
        if self.time_step_ratio <= 0:
            raise ValueError(f"MPC采样时间({self.mpc_dt}s)必须大于等于模型时间步长({self.model_dt}s)")
        
        # 打印时间步长信息
        print(f"[MPC] 时间步长配置:")
        print(f"  模型时间步长: {self.model_dt*1000:.1f}ms")
        print(f"  MPC采样时间: {self.mpc_dt*1000:.1f}ms")
        print(f"  时间步长比例: {self.time_step_ratio}")
        print(f"  预测时域: {self.Np}步 = {self.Np * self.mpc_dt:.2f}s")
        print(f"  控制时域: {self.Nc}步 = {self.Nc * self.mpc_dt:.2f}s")
        
        # 默认权重矩阵
        if Q is None:
            Q = np.diag([20, 1000, 1000, 1000, 20, 20])  # 默认Q矩阵（6维状态）
        if R is None:
            # 默认R矩阵（12维控制：6个转矩 + 6个转向角）
            # 转矩权重较小（允许较大变化），转向角权重较大（限制变化）
            R = np.diag([5, 5, 5, 5, 5, 5, 10000, 10000, 10000, 10000, 10000, 10000])
        if delta_umax is None:
            # 默认控制增量约束（12维）
            # 转矩增量：0.5（归一化后），转向角增量：0.2（归一化后）
            delta_umax = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        
        self.Q = Q
        self.R = R
        self.delta_umax = delta_umax
        
        # 构建扩展状态空间模型
        self._build_extended_model()
        
        # 预计算预测矩阵
        self._precompute_prediction_matrices()
        
        # 构建约束
        self._build_constraints()
        
        # 初始化求解器（延迟加载）
        self.solver = None
        
        # 打印跟踪误差标志（默认False，可通过enable_tracking_error_print启用）
        self._print_tracking_error = False
        
        # 检查可用的求解器
        self._check_available_solvers()
    
    def _build_extended_model(self):
        """
        构建扩展状态空间模型（包含历史控制输入）
        考虑时间步长缩放：如果MPC采样时间是模型时间步长的n倍，需要提升A和B矩阵
        """
        # self.deepedmd.A 和 self.deepedmd.B 已经在 _load_pytorch_model 中转置过了
        # 所以这里直接使用，不需要再次转置
        a = self.deepedmd.A  # (Nx, Nx) - 模型时间步长的A矩阵
        b = self.deepedmd.B  # (Nx, Nu) - 模型时间步长的B矩阵
        
        # ========== 时间步长缩放处理 ==========
        # 如果MPC采样时间是模型时间步长的n倍，需要提升A和B矩阵
        n = self.time_step_ratio
        
        if n == 1:
            # 时间步长相同，直接使用
            a_mpc = a
            b_mpc = b
        else:
            # 提升A矩阵：A_mpc = A^n
            a_mpc = np.linalg.matrix_power(a, n)
            
            # 提升B矩阵：B_mpc = sum_{i=0}^{n-1} A^i @ B
            # 这考虑了在n个模型时间步长内，控制输入的累积效应
            b_mpc = np.zeros_like(b)
            a_power = np.eye(a.shape[0])  # A^0 = I
            for i in range(n):
                b_mpc += a_power @ b
                if i < n - 1:  # 最后一次不需要再计算A的幂
                    a_power = a_power @ a
        
        # 扩展状态: [x_lift; u_prev]
        # 扩展系统: [x_lift_{k+1}; u_k] = A_ext * [x_lift_k; u_{k-1}] + B_ext * delta_u_k
        # 注意：这里使用提升后的A_mpc和B_mpc，使得每个MPC步对应正确的采样时间
        A_ext = np.zeros((self.Nx + self.Nu, self.Nx + self.Nu))
        A_ext[:self.Nx, :self.Nx] = a_mpc  # 使用提升后的A矩阵
        A_ext[:self.Nx, self.Nx:] = b_mpc  # 使用提升后的B矩阵
        A_ext[self.Nx:, self.Nx:] = np.eye(self.Nu)
        
        B_ext = np.zeros((self.Nx + self.Nu, self.Nu))
        B_ext[:self.Nx, :] = b_mpc  # 使用提升后的B矩阵
        B_ext[self.Nx:, :] = np.eye(self.Nu)
        
        self.A_ext = A_ext
        self.B_ext = B_ext
        
        # 输出矩阵C（只输出原始状态）
        # C = [eye(state_dim) zeros(state_dim, space_dim - state_dim + Nu)]
        # 其中 space_dim - state_dim + Nu = (Nx - state_dim) + Nu
        # kesi维度是 (Nx + Nu)，C选择前state_dim维（原始状态），后面全0
        self.C = np.hstack([np.eye(self.deepedmd.s_dim), 
                           np.zeros((self.deepedmd.s_dim, self.Nx - self.deepedmd.s_dim + self.Nu))])
    
    def _precompute_prediction_matrices(self):
        """
        预计算预测矩阵PHI和THETA
        注意：A_ext和B_ext已经在_build_extended_model中提升到MPC采样时间，
        所以这里直接使用j步（而不是j * time_step_ratio步）
        """
        # PHI: 自由响应矩阵
        # PHI[j] = C @ A_ext^j，表示j个MPC步后的自由响应
        # 由于A_ext已经是提升后的矩阵（对应MPC采样时间），所以直接使用j
        PHI_list = []
        for j in range(1, self.Np + 1):
            PHI_list.append(self.C @ np.linalg.matrix_power(self.A_ext, j))
        self.PHI = np.vstack(PHI_list)  # (Np*state_dim) x (Nx+Nu)
        
        # THETA: 强制响应矩阵
        # THETA[j,k] = C @ A_ext^(j-k) @ B_ext，表示在第k步施加控制输入，在第j步的响应
        # 由于A_ext和B_ext已经是提升后的矩阵，所以直接使用(j-k)
        THETA_list = []
        for j in range(1, self.Np + 1):
            row = []
            for k in range(1, self.Nc + 1):
                if k <= j:
                    row.append(self.C @ np.linalg.matrix_power(self.A_ext, j - k) @ self.B_ext)
                else:
                    row.append(np.zeros((self.deepedmd.s_dim, self.Nu)))
            THETA_list.append(np.hstack(row))
        self.THETA = np.vstack(THETA_list)  # (Np*state_dim) x (Nc*Nu)
    
    def _build_constraints(self):
        """构建约束矩阵，并预计算H和A_combined矩阵"""
        # 控制输入约束矩阵A_l（累积约束）
        A_l = np.zeros((self.Nc, self.Nc))
        for p in range(self.Nc):
            for q in range(self.Nc):
                if q <= p:
                    A_l[p, q] = 1
        
        self.A_l = np.kron(A_l, np.eye(self.Nu))  # (Nc*Nu) x (Nc*Nu)
        
        # 控制输入边界（归一化范围[0, 1]）
        umin = np.zeros(self.Nu)  # [0, 0, ..., 0]
        umax = np.ones(self.Nu)   # [1, 1, ..., 1]
        self.Umin = np.kron(np.ones(self.Nc), umin)
        self.Umax = np.kron(np.ones(self.Nc), umax)
        
        # 控制增量边界
        delta_umin = -self.delta_umax
        self.delta_Umin = np.kron(np.ones(self.Nc), delta_umin)
        self.delta_Umax = np.kron(np.ones(self.Nc), self.delta_umax)
        
        # ========== 优化1: 预计算H矩阵（Hessian矩阵） ==========
        # H矩阵的结构是固定的，只依赖于THETA、Q、R
        # 注意：THETA是在_precompute_prediction_matrices()中计算的，所以在这里可以安全地使用
        Q_kron = np.kron(np.eye(self.Np), self.Q)
        R_kron = np.kron(np.eye(self.Nc), self.R)
        H_11 = 2 * (self.THETA.T @ Q_kron @ self.THETA) + R_kron
        H_12 = np.zeros((self.Nc * self.Nu, 1))
        H_22 = np.array([[100.0]])  # 松弛因子
        H = np.block([[H_11, H_12], [H_12.T, H_22]])
        H = (H + H.T) / 2  # 确保对称
        
        # 添加数值稳定性：对H矩阵添加正则化项，避免病态问题
        # 这可以防止在边界情况下H矩阵条件数过大
        # 短期优化：从1e-8增加到1e-6，改善H矩阵条件数
        reg_factor = 1e-6
        H = H + reg_factor * np.eye(H.shape[0])
        
        # 同时存储密集矩阵（用于quadprog）和稀疏矩阵（用于OSQP）
        from scipy import sparse
        self.H_dense = H  # 存储密集矩阵用于quadprog
        self.H_sparse = sparse.csc_matrix(H)  # 存储稀疏矩阵用于OSQP
        
        # 存储Q_kron用于后续计算f向量
        self.Q_kron = Q_kron
        
        # ========== 优化2: 预计算A_combined矩阵 ==========
        # A_combined的结构是固定的，只依赖于A_l
        n_vars = self.Nc * self.Nu + 1  # 变量数量：[delta_U; slack]
        
        # 构建A_ineq（结构固定）
        A_ineq_1 = np.hstack([self.A_l, np.zeros((self.Nc * self.Nu, 1))])
        A_ineq_2 = np.hstack([-self.A_l, np.zeros((self.Nc * self.Nu, 1))])
        A_ineq = np.vstack([A_ineq_1, A_ineq_2])
        
        # 转换为稀疏矩阵
        A_ineq_sparse = sparse.csc_matrix(A_ineq)
        I_sparse = sparse.eye(n_vars, format='csc')
        
        # 合并约束矩阵并存储
        self.A_combined = sparse.vstack([A_ineq_sparse, I_sparse], format='csc')
        
        # 存储n_vars用于后续使用
        self.n_vars = n_vars
        
        # 预计算lb和ub的固定部分（边界约束是固定的）
        self.lb_fixed = np.concatenate([self.delta_Umin, [0.0]])
        self.ub_fixed = np.concatenate([self.delta_Umax, [0.0]])
        
        # 检查可用的求解器
        self._check_available_solvers()
    
    def _check_available_solvers(self):
        """检查可用的求解器并打印信息"""
        available_solvers = []
        
        # 检查quadprog
        try:
            import quadprog
            available_solvers.append("quadprog (推荐)")
        except ImportError:
            pass
        
        # 检查OSQP
        try:
            import osqp
            available_solvers.append("OSQP")
        except ImportError:
            pass
        
        # scipy.optimize暂不使用
        # available_solvers.append("scipy.optimize (备选)")
        
        if len(available_solvers) > 0:
            print(f"[MPC] 可用求解器: {', '.join(available_solvers)}")
            if "quadprog" not in str(available_solvers[0]):
                print(f"[MPC] 警告: 建议安装quadprog以获得最佳稳定性: pip install quadprog")
        else:
            print(f"[MPC] 警告: 没有可用的QP求解器！")
    
    def solve(self, x_lift: np.ndarray, u_prev: np.ndarray, ref_traj: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        求解MPC优化问题
        
        Args:
            x_lift: 当前提升状态 [space_dim]
            u_prev: 上一时刻控制输入 [control_dim]
            ref_traj: 参考轨迹 [Np, state_dim]（归一化后的局部状态）
        
        Returns:
            delta_u: 控制增量 [control_dim]
            success: 是否求解成功
        """
        # 构建扩展状态
        kesi = np.concatenate([x_lift, u_prev])
        
        # 构建参考轨迹向量（对应MATLAB: reshape(ref_r', [state_dim * Np, 1])）
        # ref_traj是(Np, state_dim)，转置后是(state_dim, Np)，按列优先展开成(state_dim*Np,)或按行优先展开成(Np*state_dim,)
        # MATLAB的reshape默认是列优先，所以reshape(ref_r', [state_dim*Np, 1])等价于按列展开
        # 但ref_r'是(state_dim, Np)，按列展开就是先取第1列，再取第2列...
        # Python的flatten默认按行优先，对应MATLAB的reshape(ref_r, [1, Np*state_dim])
        # 为了匹配MATLAB，我们需要转置后按列优先展开
        ref_r = ref_traj.T.flatten('F')  # 转置后按列优先展开，对应MATLAB的reshape(ref_r', [...])
        
        # 计算跟踪误差
        error = self.PHI @ kesi - ref_r
        
        # 打印跟踪误差信息（可选，通过类属性控制）
        if hasattr(self, '_print_tracking_error') and self._print_tracking_error:
            # 计算误差统计
            error_norm = np.linalg.norm(error)
            error_max = np.max(np.abs(error))
            error_mean = np.mean(np.abs(error))
            
            # 将误差reshape为(Np, state_dim)以便分析各维度误差
            error_reshaped = error.reshape(self.Np, self.deepedmd.s_dim)
            error_by_dim = np.mean(np.abs(error_reshaped), axis=0)  # 各维度平均误差
            
            print(f"[MPC跟踪误差]")
            print(f"  误差范数: {error_norm:.6f}, 最大误差: {error_max:.6f}, 平均误差: {error_mean:.6f}")
            print(f"  各维度平均误差: x={error_by_dim[0]:.6f}, y={error_by_dim[1]:.6f}, yaw={error_by_dim[2]:.6f}, "
                  f"vx={error_by_dim[3]:.6f}, vy={error_by_dim[4]:.6f}, yaw_rate={error_by_dim[5]:.6f}")
        
        # 构建QP问题: min 0.5 * x^T * H * x + f^T * x
        # 其中 x = [delta_U; slack]
        
        # ========== 使用预计算的H矩阵（优化1） ==========
        # H矩阵已经在_build_constraints()中预计算并存储为self.H_sparse
        
        # 梯度向量（每次需要重新计算，因为依赖于error）
        # MATLAB: f_cell{1,1} = 2*error'*Q*THETA; f = cell2mat(f_cell);
        # error是(state_dim*Np,)列向量，error.T是(1, state_dim*Np)行向量
        # error.T @ Q_kron @ THETA 结果是(1, Nu*Nc)行向量
        # 但quadprog需要列向量，所以需要转置
        f_1 = (2 * error.T @ self.Q_kron @ self.THETA).flatten()  # (Nu*Nc,)
        f_2 = np.array([0.0])  # 松弛变量系数
        f = np.concatenate([f_1, f_2])  # (Nu*Nc+1,)列向量
        
        # ========== 约束: A_ineq * x <= b_ineq ==========
        # A_combined已经在_build_constraints()中预计算（优化2）
        # 只需要计算b_ineq（依赖于u_prev，每次都在变化）
        Ut = np.kron(np.ones(self.Nc), u_prev)
        b_ineq_1 = self.Umax - Ut
        b_ineq_2 = -self.Umin + Ut
        b_ineq = np.concatenate([b_ineq_1, b_ineq_2])
        
        # ========== 边界约束 ==========
        # 动态调整delta_u的边界，避免约束不可行
        # 当u_prev接近边界时，限制delta_u的方向
        lb = self.lb_fixed.copy()
        ub = self.ub_fixed.copy()
        
        # 检查并调整边界，确保约束可行
        # 对于每个控制输入维度，需要调整所有Nc步中该维度的边界
        # 归一化范围[0, 1]
        for i in range(self.Nu):
            u_prev_i = u_prev[i]
            
            # 如果u_prev接近下界，限制delta_u不能为负（或只能很小的负值）
            if u_prev_i <= 0.05:  # 接近下界0
                # delta_u[i]必须 >= -u_prev_i，确保u_new >= 0
                min_delta = -u_prev_i + 1e-6  # 添加小的安全裕度
                # 调整所有Nc步中第i个控制输入维度的下界
                for k in range(self.Nc):
                    idx = k * self.Nu + i
                    lb[idx] = max(lb[idx], min_delta)
            # 如果u_prev接近上界，限制delta_u不能为正
            elif u_prev_i >= 0.95:  # 接近上界1
                # delta_u[i]必须 <= (1 - u_prev_i)，确保u_new <= 1
                max_delta = (1.0 - u_prev_i) - 1e-6  # 添加小的安全裕度
                # 调整所有Nc步中第i个控制输入维度的上界
                for k in range(self.Nc):
                    idx = k * self.Nu + i
                    ub[idx] = min(ub[idx], max_delta)
        
        # 确保lb <= ub（避免无效约束）
        for i in range(len(lb)):
            if lb[i] > ub[i] + 1e-10:  # 添加容差，避免数值误差
                # 如果约束无效，调整到合理范围
                mid = (lb[i] + ub[i]) / 2
                lb[i] = mid - 1e-6
                ub[i] = mid + 1e-6
                
                # 对于控制增量维度（前Nc*Nu个元素），确保在合理范围内
                if i < self.Nc * self.Nu:
                    # 确定是哪个控制输入维度
                    dim_idx = i % self.Nu
                    # 控制增量维度，限制在[-delta_umax, delta_umax]内
                    lb[i] = max(lb[i], -self.delta_umax[dim_idx])
                    ub[i] = min(ub[i], self.delta_umax[dim_idx])
        
        # 求解QP问题
        # 使用多级fallback策略：OSQP -> quadprog
        # 注意：scipy.optimize暂不使用
        
        # ========== 方法1: 优先使用OSQP（高性能QP求解器） ==========
        try:
            import osqp
            
            # OSQP标准形式: min 0.5 * x^T * P * x + q^T * x, s.t. l <= A*x <= u
            # 我们需要将约束 A_ineq * x <= b_ineq 转换为 l <= A*x <= u
            # 其中 A = [A_ineq; I], l = [-inf; lb], u = [b_ineq; ub]
            
            # ========== 使用预计算的A_combined矩阵（优化2） ==========
            # A_combined已经在_build_constraints()中预计算并存储为self.A_combined
            
            # 构建l和u向量（使用大数值代替-inf，OSQP支持inf但更新时可能有问题）
            # 对于不等式约束 A_ineq * x <= b_ineq，使用 -1e20 代替 -inf
            LARGE_NEG = -1e20
            l_combined = np.concatenate([LARGE_NEG * np.ones(len(b_ineq)), lb])
            u_combined = np.concatenate([b_ineq, ub])
            
            # 创建或更新求解器
            if self.solver is None:
                self.solver = osqp.OSQP()
                # 设置求解器参数（与MATLAB quadprog类似）
                settings = {
                    'verbose': False,
                    'eps_abs': 1e-5,  # 绝对容差
                    'eps_rel': 1e-5,  # 相对容差
                    'max_iter': 4000,  # 最大迭代次数（MATLAB使用非常大的值）
                    'warm_start': True,  # 启用热启动
                    'polish': True,  # 启用精炼步骤
                }
                self.solver.setup(P=self.H_sparse, q=f, A=self.A_combined, l=l_combined, u=u_combined, **settings)
            else:
                # 更新问题（热启动，只更新可能变化的参数）
                try:
                    self.solver.update(q=f, l=l_combined, u=u_combined)
                except Exception:
                    # 如果更新失败，重新设置求解器（这种情况很少发生）
                    settings = {
                        'verbose': False,
                        'eps_abs': 1e-5,
                        'eps_rel': 1e-5,
                        'max_iter': 4000,
                        'warm_start': True,
                        'polish': True,
                    }
                    self.solver.setup(P=self.H_sparse, q=f, A=self.A_combined, l=l_combined, u=u_combined, **settings)
            
            # 求解
            result = self.solver.solve()
            
            # 检查求解状态
            if result.info.status in ['solved', 'solved inaccurate']:
                X = result.x
                delta_u = X[:self.Nu]
                return delta_u, True
            else:
                # OSQP失败，尝试quadprog
                print(f"[MPC] OSQP status={result.info.status}, trying quadprog...")
                raise ValueError(f"OSQP status: {result.info.status}")
                
        except (ImportError, ValueError, Exception) as e:
            # OSQP不可用或失败，尝试quadprog
            if isinstance(e, ImportError):
                print(f"[MPC] OSQP not available (ImportError), trying quadprog...")
            else:
                print(f"[MPC] OSQP failed ({e}), trying quadprog...")
            
            # ========== 方法2: 使用quadprog（备选QP求解器） ==========
            try:
                import quadprog
                
                # quadprog标准形式: min 0.5 * x^T * G * x + a^T * x, s.t. C^T * x >= b
                # 我们需要将问题转换为quadprog格式
                # 约束: A_ineq * x <= b_ineq 等价于 -A_ineq * x >= -b_ineq
                # 边界: lb <= x <= ub 等价于 x >= lb 和 -x >= -ub
                
                # 使用预存储的密集矩阵（quadprog需要）
                H_dense = self.H_dense
                
                # 构建约束矩阵 C^T * x >= b
                # 约束1: A_ineq * x <= b_ineq  -> -A_ineq * x >= -b_ineq
                A_ineq_1 = np.hstack([self.A_l, np.zeros((self.Nc * self.Nu, 1))])
                A_ineq_2 = np.hstack([-self.A_l, np.zeros((self.Nc * self.Nu, 1))])
                A_ineq = np.vstack([A_ineq_1, A_ineq_2])
                
                # 约束2: x >= lb  -> I * x >= lb
                # 约束3: x <= ub  -> -I * x >= -ub
                n_vars = len(f)
                I_mat = np.eye(n_vars)
                
                # 合并所有约束: C^T * x >= b
                # quadprog的API: solve_qp(G, a, C, b, meq=0)
                # 其中C的形状是 (n_vars, n_constraints)，表示 C^T * x >= b
                # 所以我们需要构建C使得 C^T * x >= b
                C_constraints = np.vstack([-A_ineq, I_mat, -I_mat])  # (n_constraints, n_vars)
                b_constraints = np.concatenate([-b_ineq, lb, -ub])  # (n_constraints,)
                
                # quadprog要求C的形状是 (n_vars, n_constraints)
                # 所以需要转置：C = C_constraints^T
                C_T = C_constraints.T  # (n_vars, n_constraints)
                
                # 检查约束可行性（避免数值问题）
                # 注意：直接限制b_constraints可能会改变约束语义，导致不可行
                # 如果b_constraints中有很大的负数，可能是约束本身不可行
                # 这里只对极端值进行限制，避免数值溢出，但不改变约束的可行性
                b_safe = b_constraints.copy()
                # 只对极端负值进行限制（避免数值溢出），但保持约束的语义
                extreme_neg_mask = b_safe < -1e10
                if np.any(extreme_neg_mask):
                    # 如果约束本身不可行，应该让求解器报告，而不是强制限制
                    # 但为了避免数值溢出，可以设置一个合理的下界
                    b_safe[extreme_neg_mask] = -1e10
                
                # 调用quadprog求解
                # quadprog.solve_qp(G, a, C, b, meq=0)
                # 注意：quadprog的G是Hessian矩阵（不需要乘以2），a是梯度向量
                # C的形状应该是 (n_constraints, n_vars)，b是 (n_constraints,)
                try:
                    # 确保H矩阵是正定的（添加小的正则化项）
                    H_quadprog = H_dense.copy()
                    
                    # 检查H矩阵的条件数和特征值
                    try:
                        H_cond = np.linalg.cond(H_quadprog)
                        H_eigvals = np.linalg.eigvals(H_quadprog)
                        min_eigval = np.min(H_eigvals)
                        
                        # 如果条件数过大或最小特征值过小，增强正则化
                        reg_factor = 1e-8  # 基础正则化
                        if H_cond > 1e10:
                            # 条件数过大，根据条件数动态调整正则化
                            reg_factor = max(reg_factor, 1e-8 * (H_cond / 1e10))
                        
                        if min_eigval < 1e-8:
                            # 最小特征值过小，确保正定性
                            reg_factor = max(reg_factor, -min_eigval + 1e-6)
                        
                        if reg_factor > 1e-8:
                            H_quadprog = H_quadprog + reg_factor * np.eye(H_quadprog.shape[0])
                    except:
                        # 如果计算失败，使用基础正则化
                        H_quadprog = H_quadprog + 1e-8 * np.eye(H_quadprog.shape[0])
                    
                    # 确保C_T和b_safe的形状正确
                    # C_T应该是 (n_vars, n_constraints) - quadprog要求的格式
                    # b_safe应该是 (n_constraints,)
                    if C_T.shape[1] != len(b_safe):
                        raise ValueError(f"约束矩阵维度不匹配: C_T.shape={C_T.shape}, b_safe.shape={b_safe.shape}, 期望C_T.shape[1]==len(b_safe)")
                    
                    # quadprog.solve_qp返回格式: (x, fval, u, iact, nact, iter)
                    # x: 解向量
                    # fval: 目标函数值
                    # u: 拉格朗日乘数
                    # iact: 活动约束索引
                    # nact: 活动约束数量
                    # iter: 迭代次数（>=0表示成功）
                    result = quadprog.solve_qp(H_quadprog, f, C_T, b_safe, meq=0)
                    
                    # 将结果转换为元组（如果不是元组）
                    if not isinstance(result, tuple):
                        result = (result,)
                    
                    # 提取值
                    if len(result) >= 2:
                        X = result[0]  # 解向量
                        fval = result[1]  # 目标函数值
                        
                        # 判断是否成功：通常通过迭代次数或解的有效性
                        if len(result) >= 6:
                            iter_count = result[5]  # 迭代次数
                            # 处理iter_count可能是数组的情况
                            try:
                                if isinstance(iter_count, np.ndarray):
                                    # 安全地获取数组大小
                                    arr_size = int(iter_count.size)
                                    if arr_size == 1:
                                        iter_count = float(iter_count.item())
                                    elif arr_size > 0:
                                        iter_count = float(iter_count.flat[0])
                                    else:
                                        iter_count = -1
                                else:
                                    iter_count = float(iter_count)
                            except (ValueError, TypeError, AttributeError):
                                # 如果转换失败，假设成功
                                iter_count = 0
                            # 如果迭代次数 >= 0，通常表示成功
                            # 同时检查X是否有效
                            exitflag = 0 if iter_count >= 0 and X is not None else -1
                        else:
                            # 如果没有iter，通过检查X是否有效
                            exitflag = 0 if X is not None and len(X) > 0 else -1
                    else:
                        raise ValueError(f"quadprog返回了意外的值数量: {len(result)}")
                    
                    # 处理exitflag可能是数组的情况
                    if isinstance(exitflag, np.ndarray):
                        exitflag = exitflag.item() if exitflag.size == 1 else exitflag[0]
                    
                    if exitflag == 0:  # quadprog成功
                        delta_u = X[:self.Nu]
                        return delta_u, True
                    else:
                        # quadprog失败
                        print(f"[MPC] quadprog exitflag={exitflag}")
                        raise ValueError(f"quadprog exitflag: {exitflag}")
                except Exception as e:
                    # quadprog内部错误
                    print(f"[MPC] quadprog调用失败: {type(e).__name__}: {e}")
                    raise ValueError(f"quadprog error: {e}")
                    
            except (ImportError, ValueError) as e2:
                # quadprog也不可用或失败，返回零控制
                if isinstance(e2, ImportError):
                    print(f"[MPC] quadprog not available. 请安装OSQP或quadprog")
                else:
                    print(f"[MPC] quadprog failed: {e2}")
                print(f"[MPC] 返回零控制增量，u_prev={u_prev}")
                return np.zeros(self.Nu), False
    
    def convert_to_control_output(self, u_normalized: np.ndarray) -> np.ndarray:
        """
        将归一化控制输入转换为实际控制指令（Trucksim格式）
        
        Args:
            u_normalized: 归一化控制输入 [12维]，范围[0, 1]（使用CONTROL_MIN/MAX归一化）
        
        Returns:
            control: [12维]
                    [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
                     steer_LR(deg), steer_RR(deg),
                     torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
                     torque_LR(N·m), torque_RR(N·m)]
        
        Raises:
            ValueError: 如果输入维度不正确
        """
        # 确保输入是12维
        if len(u_normalized) != 12:
            raise ValueError(f"控制输入维度错误: 期望12维，实际{len(u_normalized)}维")
        
        # 反归一化控制输入（使用CONTROL_MIN/MAX反归一化）
        u_denorm = self.deepedmd.denormalize_control(u_normalized)
        
        # 提取转矩和转向角
        # u_denorm格式：[torque_LF, torque_RF, torque_LM, torque_RM, torque_LR, torque_RR,
        #               steer_LF, steer_RF, steer_LM, steer_RM, steer_LR, steer_RR]
        # 注意：inference_demo.py中控制输入格式是：6个转矩 + 6个转向角
        torques = u_denorm[:6]  # 前6维：转矩 (N·m)
        steer_angles = u_denorm[6:12]  # 后6维：转向角 (rad)
        
        # 转换为Trucksim输出格式：
        # [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
        #  steer_LR(deg), steer_RR(deg),
        #  torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
        #  torque_LR(N·m), torque_RR(N·m)]
        steer_angles_deg = steer_angles * 180 / np.pi  # 转换为度
        
        return np.array([
            steer_angles_deg[0],  # steer_LF
            steer_angles_deg[1],  # steer_RF
            steer_angles_deg[2],  # steer_LM
            steer_angles_deg[3],  # steer_RM
            steer_angles_deg[4],  # steer_LR
            steer_angles_deg[5],  # steer_RR
            torques[0],  # torque_LF
            torques[1],  # torque_RF
            torques[2],  # torque_LM
            torques[3],  # torque_RM
            torques[4],  # torque_LR
            torques[5]   # torque_RR
        ])

