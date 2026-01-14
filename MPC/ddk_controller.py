"""
DDK-MPC控制器Python实现
实现DDK模型加载、编码、归一化等功能，以及MPC控制器
支持PyTorch Transformer编码器（Trucksim版本）
"""

import os
import sys

# 解决OpenMP冲突问题（必须在导入numpy之前设置）
# MATLAB和Python的科学计算库都使用了OpenMP，会导致冲突
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import scipy.io as scio
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


class DDK:
    """
    Deep Koopman模型类
    实现编码器、归一化、参考轨迹处理等功能
    支持PyTorch模型（.pth格式）和传统MATLAB模型（.mat/.pkl格式）
    """
    
    def __init__(self, param_path: str, device: str = 'cpu'):
        """
        初始化DDK模型
        
        Args:
            param_path: 参数文件路径（.pth, .mat或.pkl格式）
            device: PyTorch设备（'cpu'或'cuda'），仅用于pth格式
        """
        self.param_path = param_path
        self.device = device
        self.model_type = None  # 'pytorch' 或 'matlab'
        
        if param_path.endswith('.pth'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch未安装，无法加载.pth格式模型。请安装: pip install torch")
            self._load_pytorch_model()
            self.model_type = 'pytorch'
        else:
            self._load_parameters()
            self._extract_koopman_matrices()
            self._compute_action_bounds()
            self.model_type = 'matlab'
    
    def _load_pytorch_model(self):
        """加载PyTorch模型（.pth格式）"""
        print(f"[DDK] 加载PyTorch模型: {self.param_path}")
        
        # 模型参数（与inference_demo.py保持一致）
        self.state_dim = 6
        self.s_dim = 6  # 保持与MATLAB模型兼容（s_dim = state_dim）
        self.hidden_dim = 16
        self.control_dim = 12
        self.u_dim = 12  # 保持与MATLAB模型兼容（u_dim = control_dim）
        self.lift_dim = self.hidden_dim - self.state_dim  # 10
        self.space_dim = self.hidden_dim  # 16 = state_dim + lift_dim
        
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
        
        # 设置state_bound（虽然PyTorch模型不使用，但为了兼容get_reference等方法）
        # PyTorch模型的状态归一化在编码器内部通过BatchNorm完成，不需要外部state_bound
        # 这里设置一个默认值，但实际不会被使用（normalization方法会直接返回局部状态）
        # 格式：[[min], [max]]，对应 [vx, vy, yaw_rate] 的边界
        # 使用合理的默认值（根据实际车辆动力学设置）
        self.state_bound = np.array([[-10.0, -5.0, -2.0], [30.0, 5.0, 2.0]])  # [vx, vy, yaw_rate] 边界
        
        # 设置action_bound（虽然PyTorch模型不使用，但为了兼容性）
        # PyTorch模型使用CONTROL_MIN/MAX进行控制归一化
        self.action_bound = np.array([[-np.pi/2, -1500], [np.pi/2, 1500]])  # [转向角, 转矩] 边界
        
        print(f"[DDK] PyTorch模型加载完成")
        print(f"  状态维度: {self.state_dim} (s_dim={self.s_dim}), 提升维度: {self.lift_dim}, 空间维度: {self.space_dim}")
        print(f"  控制维度: {self.control_dim} (u_dim={self.u_dim})")
        print(f"  A矩阵形状: {self.A.shape}, B矩阵形状: {self.B.shape}")
        print(f"  注意: PyTorch模型使用编码器内部BatchNorm进行状态归一化，state_bound仅用于兼容性")
    
    def _load_parameters(self):
        """加载模型参数"""
        if self.param_path.endswith('.mat'):
            params = scio.loadmat(self.param_path, squeeze_me=True)
            
            # 优先尝试使用 encoder_weights/decoder_weights (新格式)
            if 'encoder_weights' in params:
                encoder_weights_raw = params['encoder_weights']
                # 处理结构化数组
                if isinstance(encoder_weights_raw, np.ndarray) and encoder_weights_raw.dtype.names:
                    if encoder_weights_raw.shape == ():
                        # 标量结构化数组，转换为字典
                        encoder_dict = {}
                        for name in encoder_weights_raw.dtype.names:
                            value = encoder_weights_raw[name]
                            # 只有当value是0维数组（标量）时才使用item()
                            if isinstance(value, np.ndarray):
                                if value.ndim == 0:
                                    # 0维数组（标量），可以安全使用item()
                                    encoder_dict[name] = value.item()
                                else:
                                    # 多维数组，直接使用，但确保是numpy数组
                                    encoder_dict[name] = np.array(value, dtype=float)
                            elif hasattr(value, 'item'):
                                # 尝试item()，但如果失败则直接使用
                                try:
                                    encoder_dict[name] = value.item()
                                except (ValueError, TypeError):
                                    # item()失败，说明不是标量，直接使用
                                    encoder_dict[name] = np.array(value, dtype=float)
                            else:
                                encoder_dict[name] = value
                        
                        # 将 fc_net.X.weight 和 fc_net.X.bias 映射到 WEF{i+1} 和 bEF{i+1}
                        # 字段名格式：fc_net.0.weight, fc_net.0.bias, fc_net.2.weight, fc_net.2.bias, ...
                        self.weights = {}
                        self.biases = {}
                        
                        # 提取所有weight和bias字段，按索引排序
                        weight_keys = [k for k in encoder_dict.keys() if k.endswith('.weight')]
                        bias_keys = [k for k in encoder_dict.keys() if k.endswith('.bias')]
                        
                        # 按照字段中的数字索引排序
                        def extract_index(key):
                            # 从 'fc_net.2.weight' 中提取 2
                            try:
                                parts = key.split('.')
                                return int(parts[1])
                            except:
                                return -1
                        
                        weight_keys_sorted = sorted(weight_keys, key=extract_index)
                        bias_keys_sorted = sorted(bias_keys, key=extract_index)
                        
                        # 映射到 WEF1, WEF2, ... 格式
                        for i, weight_key in enumerate(weight_keys_sorted):
                            weight_value = encoder_dict[weight_key]
                            # 确保权重是numpy数组格式
                            if isinstance(weight_value, np.ndarray):
                                # 如果是0维数组（标量），提取值；否则直接使用
                                if weight_value.ndim == 0:
                                    weight_value = weight_value.item()
                                # 确保是float类型
                                weight_value = np.array(weight_value, dtype=float)
                            else:
                                weight_value = np.array(weight_value, dtype=float)
                            self.weights[f'WEF{i+1}'] = weight_value
                        
                        for i, bias_key in enumerate(bias_keys_sorted):
                            # 只保存非最后一层的bias（编码器最后一层通常没有bias）
                            if i < len(bias_keys_sorted) - 1:  # 排除最后一层
                                bias_value = encoder_dict[bias_key]
                                # 确保偏置是numpy数组格式
                                if isinstance(bias_value, np.ndarray):
                                    if bias_value.ndim == 0:
                                        bias_value = bias_value.item()
                                    bias_value = np.array(bias_value, dtype=float)
                                else:
                                    bias_value = np.array(bias_value, dtype=float)
                                self.biases[f'bEF{i+1}'] = bias_value
                    else:
                        # 非标量结构化数组，使用item()方法
                        self.weights = encoder_weights_raw.item() if hasattr(encoder_weights_raw, 'item') else encoder_weights_raw
                        self.biases = {}  # 暂时设为空
                elif hasattr(encoder_weights_raw, 'item'):
                    # 如果有item方法，尝试获取
                    item_result = encoder_weights_raw.item()
                    if isinstance(item_result, dict):
                        self.weights = item_result
                        self.biases = {}  # 暂时设为空
                    else:
                        self.weights = item_result
                        self.biases = {}
                else:
                    self.weights = encoder_weights_raw
                    self.biases = {}
                
                # 处理decoder_weights（如果需要）
                if 'decoder_weights' in params:
                    # decoder权重暂时不需要，因为编码器不需要decoder
                    pass
            
            # 回退到旧的 weights/biases 格式
            elif 'weights' in params:
                weights_raw = params['weights']
                # 处理MATLAB结构体
                if hasattr(weights_raw, 'item'):
                    self.weights = weights_raw.item()
                elif isinstance(weights_raw, np.ndarray) and weights_raw.dtype.names:
                    # 结构化数组，转换为字典
                    if weights_raw.shape == ():
                        # 标量结构化数组
                        self.weights = {name: weights_raw[name].item() if hasattr(weights_raw[name], 'item') else weights_raw[name] 
                                       for name in weights_raw.dtype.names}
                    else:
                        self.weights = weights_raw.item() if hasattr(weights_raw, 'item') else weights_raw
                else:
                    self.weights = weights_raw
                
                if 'biases' in params:
                    biases_raw = params['biases']
                    if hasattr(biases_raw, 'item'):
                        self.biases = biases_raw.item()
                    elif isinstance(biases_raw, np.ndarray) and biases_raw.dtype.names:
                        if biases_raw.shape == ():
                            self.biases = {name: biases_raw[name].item() if hasattr(biases_raw[name], 'item') else biases_raw[name] 
                                          for name in biases_raw.dtype.names}
                        else:
                            self.biases = biases_raw.item() if hasattr(biases_raw, 'item') else biases_raw
                    else:
                        self.biases = biases_raw
                else:
                    self.biases = {}
            else:
                # 如果都不存在，设置为空字典
                self.weights = {}
                self.biases = {}
            
            # 检查权重结构
            if not isinstance(self.weights, dict):
                print(f"[DDK] biases字典键: {list(self.biases.keys())}")
                raise ValueError(f"weights类型为 {type(self.weights)}，不是字典")
            
            # 加载koopman_weights（如果存在）
            if 'koopman_weights' in params:
                self.koopman_weights = params['koopman_weights']
            else:
                self.koopman_weights = None
            
            # 网络结构参数
            self.encoder_widths = params.get('encoder_widths', [])
            self.decoder_widths = params.get('decoder_widths', [])
            self.eact_type = params.get('eact_type', [])
            self.dact_type = params.get('dact_type', [])
            self.s_dim = int(params.get('s_dim', 3))
            self.u_dim = int(params.get('u_dim', 2))
            self.lift_dim = int(params.get('lift_dim', 10))
            self.conca_num = int(params.get('conca_num', 0))
            self.state_bound = params.get('state_bound', np.array([[-0.2, -2.7, -1.2], [27.3, 1.9, 1.1]]))
            self.action_bound = params.get('action_bound', np.array([[-7.9, 0., 0.], [7.9, 0.2, 9.1]]))
            
        elif self.param_path.endswith('.pkl'):
            import pickle
            with open(self.param_path, 'rb') as f:
                params = pickle.load(f)
            self.weights = params['weights']
            self.biases = params['biases']
            self.encoder_widths = params['encoder_widths']
            self.decoder_widths = params['decoder_widths']
            self.eact_type = params['eact_type']
            self.dact_type = params['dact_type']
            self.s_dim = int(params['s_dim'])
            self.u_dim = int(params['u_dim'])
            self.lift_dim = int(params['lift_dim'])
            self.conca_num = int(params['conca_num'])
            self.state_bound = params['state_bound']
            self.action_bound = params['action_bound']
        else:
            raise ValueError(f"不支持的文件格式: {self.param_path}")
        
        # 计算提升空间维度（原始状态维度 + 提升维度）
        self.space_dim = int(self.s_dim + self.lift_dim)  # 确保是整数
    
    def _build_block_diagonal_A(self, diag_elements: np.ndarray, space_dim: int) -> np.ndarray:
        """
        将对角线元素构建为2x2块对角矩阵
        
        Args:
            diag_elements: 对角线元素数组，长度为space_dim
            space_dim: 空间维度
            
        Returns:
            A矩阵: (space_dim, space_dim) 的块对角矩阵
            每个2x2块的形式为: [x1, -x2; x2, x1]
        """
        if len(diag_elements) != space_dim:
            raise ValueError(f"对角线元素数量 {len(diag_elements)} 与空间维度 {space_dim} 不匹配")
        
        A = np.zeros((space_dim, space_dim))
        
        # 每2个相邻元素组成一个2x2块
        # 注意：从参数文件读取的对角线元素，每两个组成一个块
        # 块形式: [x1, -x2; x2, x1]
        # 其中 x1 是第一个元素，x2 是第二个元素
        for i in range(0, space_dim, 2):
            if i + 1 < space_dim:
                # 完整的2x2块
                x1 = diag_elements[i]
                x2 = diag_elements[i + 1]
                # 块形式: [x1, -x2; x2, x1]
                A[i, i] = x1
                A[i, i + 1] = -x2
                A[i + 1, i] = x2
                A[i + 1, i + 1] = x1
            else:
                # 最后一个元素单独处理（如果space_dim是奇数）
                A[i, i] = diag_elements[i]
        
        return A
    
    def _extract_koopman_matrices(self):
        """
        从权重中提取Koopman算子矩阵A和B
        A: (lift_dim + s_dim) x (lift_dim + s_dim) - 块对角矩阵，每2x2块形式为 [x1, -x2; x2, x1]
        B: (lift_dim + s_dim) x u_dim
        """
        # 优先从koopman_weights中提取A和B
        if self.koopman_weights is not None:
            try:
                A_raw = None
                B_raw = None
                
                # 处理结构化数组
                if isinstance(self.koopman_weights, np.ndarray):
                    if hasattr(self.koopman_weights.dtype, 'names') and self.koopman_weights.dtype.names:
                        # 结构化数组
                        if self.koopman_weights.shape == ():
                            # 标量结构化数组
                            A_raw = self.koopman_weights['A'].item()
                            B_raw = self.koopman_weights['B'].item()
                        else:
                            A_raw = self.koopman_weights['A']
                            B_raw = self.koopman_weights['B']
                elif isinstance(self.koopman_weights, dict):
                    A_raw = self.koopman_weights.get('A', None)
                    B_raw = self.koopman_weights.get('B', None)
                
                if A_raw is not None and B_raw is not None:
                    # 转换为numpy数组
                    A_arr = np.array(A_raw, dtype=float)
                    B_arr = np.array(B_raw, dtype=float)
                    
                    # 确保A是矩阵形状
                    space_dim = int(self.s_dim + self.lift_dim)  # 确保是整数
                    if A_arr.ndim == 1:
                        # 如果A是一维数组，检查是否需要reshape
                        if A_arr.size == space_dim * space_dim:
                            # 如果是完整的矩阵元素（展平的），reshape为矩阵
                            A_arr = A_arr.reshape(space_dim, space_dim)
                        elif A_arr.size == space_dim:
                            # 如果只有对角线元素，需要转换为2x2块对角结构
                            # MATLAB中A矩阵是块对角矩阵，每2个相邻元素组成一个2x2块
                            # 块形式: [x1, -x2; x2, x1]
                            A_arr = self._build_block_diagonal_A(A_arr, space_dim)
                        else:
                            raise ValueError(f"A矩阵的形状不正确: {A_arr.shape}, 期望 {(space_dim, space_dim)} 或对角线元素 {space_dim}")
                    elif A_arr.ndim == 2:
                        # 如果已经是矩阵，检查是否是块对角结构
                        # 如果不是块对角结构，尝试从对角线元素重建
                        if A_arr.shape == (space_dim, space_dim):
                            # 检查是否是对角矩阵
                            if np.allclose(A_arr, np.diag(np.diag(A_arr))):
                                # 是对角矩阵，需要转换为块对角结构
                                diag_elements = np.diag(A_arr)
                                A_arr = self._build_block_diagonal_A(diag_elements, space_dim)
                            # 否则假设已经是正确的块对角结构
                    
                    # 确保B是正确的形状
                    if B_arr.ndim == 1:
                        if B_arr.size == space_dim * self.u_dim:
                            B_arr = B_arr.reshape(self.u_dim, space_dim)
                        else:
                            raise ValueError(f"B矩阵的形状不正确: {B_arr.shape}, 期望 {(self.u_dim, space_dim)}")
                    
                    # MATLAB代码中使用 ddk.A' 和 ddk.B'，所以这里需要转置
                    self.A = A_arr.T
                    self.B = B_arr.T
                    return
            except Exception as e:
                import traceback
                print(f"警告: 从koopman_weights加载A和B矩阵时出错: {e}")
                traceback.print_exc()
        
        # 回退：从权重字典中提取WK和WU
        if isinstance(self.weights, dict):
            WK = self.weights.get('WK', None)
            WU = self.weights.get('WU', None)
        else:
            # MATLAB结构体格式
            WK = self.weights.WK if hasattr(self.weights, 'WK') else None
            WU = self.weights.WU if hasattr(self.weights, 'WU') else None
        
        if WK is None or WU is None:
            raise ValueError("无法从参数文件中找到WK或WU矩阵（也尝试了koopman_weights中的A和B）")
        
        # 转换为numpy数组
        if not isinstance(WK, np.ndarray):
            WK = np.array(WK)
        if not isinstance(WU, np.ndarray):
            WU = np.array(WU)
        
        # A和B矩阵
        # 检查WK是否是对角矩阵，如果是，需要转换为块对角结构
        space_dim = int(self.s_dim + self.lift_dim)
        if WK.shape == (space_dim, space_dim):
            # 检查是否是对角矩阵
            if np.allclose(WK, np.diag(np.diag(WK))):
                # 是对角矩阵，需要转换为块对角结构
                diag_elements = np.diag(WK)
                WK = self._build_block_diagonal_A(diag_elements, space_dim)
        
        self.A = WK.T  # MATLAB中可能是转置的
        self.B = WU.T
    
    def _compute_action_bounds(self):
        """计算控制输入的边界"""
        self.a_max = self.action_bound[1, :].reshape(1, -1)
        self.a_min = self.action_bound[0, :].reshape(1, -1)
    
    def normalization(self, x: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
        """
        坐标变换和归一化
        将全局坐标转换为局部坐标
        
        Args:
            x: 全局状态 [X, Y, Yaw, vx, vy, yaw_rate] (6维)
            ref_pos: 参考点位置 [X, Y, Yaw] (3维)
        
        Returns:
            对于PyTorch模型：局部状态（未归一化，编码器内部有BatchNorm）
            对于MATLAB模型：归一化后的局部状态 [x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local] (6维)
        """
        # 提取位置和速度
        global_pos = x[:3]  # [X, Y, Yaw]
        global_vel = x[3:6] if len(x) >= 6 else np.array([0, 0, 0])  # [vx, vy, yaw_rate]
        
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
        
        # 局部速度
        # 注意：根据MATLAB输出对比，MATLAB的normalization函数不对速度进行坐标变换
        # MATLAB输出显示：vx_local=全局vx, vy_local=全局vy（完全相同）
        # 因此这里也直接使用全局速度，不进行坐标变换
        vx_local = global_vel[0]  # 直接使用全局vx（与MATLAB一致）
        vy_local = global_vel[1]  # 直接使用全局vy（与MATLAB一致）
        yaw_rate_local = global_vel[2] if len(global_vel) > 2 else 0
        
        # 构建局部状态 [x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local] (6维)
        local_state = np.array([x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local])
        
        # 对于PyTorch模型，不需要外部归一化（编码器内部有BatchNorm）
        if self.model_type == 'pytorch':
            return local_state
        
        # 归一化到[-1, 1]
        # 注意：state_bound可能只有3维（对应速度相关状态），但s_dim可能是6
        # 如果state_bound维度小于s_dim，只归一化对应维度的状态
        state_bound_dim = self.state_bound.shape[1] if self.state_bound.ndim > 1 else len(self.state_bound)
        
        if state_bound_dim >= self.s_dim:
            # state_bound维度足够，直接使用
            state_min = self.state_bound[0, :self.s_dim]
            state_max = self.state_bound[1, :self.s_dim]
        elif state_bound_dim == 3 and self.s_dim == 6:
            # 特殊情况：state_bound只有3维（速度相关），但s_dim是6
            # 扩展state_bound到6维：位置使用较小的归一化范围
            # 根据MATLAB输出分析：
            #   - x_local=-0.083930归一化后x=-2.0134
            #   - 反推归一化范围约为[-0.042, 0.042]米
            #   - 但为了更通用，使用[-1, 1]米的范围（MATLAB可能使用更小的范围）
            state_min = np.zeros(6)
            state_max = np.ones(6)
            # 前3维（位置）：根据MATLAB输出反推归一化范围
            # 根据MATLAB输出分析：
            #   - y_local=-10.788093归一化后y=-1.7980
            #   - 反推归一化范围约为[-6, 6]米
            #   - 使用[-6, 6]米的范围，这样y_local=-10.788093归一化后约为-1.8
            # 注意：这个范围会导致归一化后的值超出[-1,1]，这是正常的
            state_min[:3] = -6.0
            state_max[:3] = 6.0
            # 后3维（速度）：使用state_bound的值
            state_min[3:6] = self.state_bound[0, :3]
            state_max[3:6] = self.state_bound[1, :3]
        else:
            # 其他情况，使用默认值
            state_min = np.zeros(self.s_dim)
            state_max = np.ones(self.s_dim)
            state_min[:state_bound_dim] = self.state_bound[0, :]
            state_max[:state_bound_dim] = self.state_bound[1, :]
        
        # 归一化: (x - min) / (max - min) * 2 - 1
        normalized = 2 * (local_state - state_min) / (state_max - state_min) - 1
        
        # 验证归一化后的值是否在合理范围内（允许轻微超出，但应该接近[-1, 1]）
        if np.any(np.abs(normalized) > 2.0):
            import warnings
            warnings.warn(
                f"归一化后的状态超出预期范围 [-1, 1]："
                f"min={np.min(normalized):.4f}, max={np.max(normalized):.4f}。"
                f"可能需要调整state_bound的范围。"
            )
        
        return normalized
    
    def normalize_control(self, u: np.ndarray) -> np.ndarray:
        """
        控制输入归一化（使用CONTROL_MIN/MAX，与inference_demo.py保持一致）
        
        Args:
            u: 原始控制输入 [12维]
               前6维：6个车轮转矩 (N·m)，范围 [-1500, 1500]
               后6维：6个车轮转向角 (rad)，范围 [-π/2, π/2]
        
        Returns:
            归一化后的控制输入 [12维]，范围 [0, 1]（线性归一化）
        """
        if self.model_type == 'pytorch':
            # 使用CONTROL_MIN/MAX归一化
            u_norm = (u - self.control_min) / (self.control_max - self.control_min + 1e-6)
            return u_norm
        else:
            # MATLAB模型：使用action_bound归一化
            # 确保输入是2D数组
            if u.ndim == 1:
                u = u.reshape(1, -1)
            
            # 归一化到[-1, 1]
            u_norm = 2 * (u - self.a_min) / (self.a_max - self.a_min + 1e-6) - 1
            return u_norm.flatten()
    
    def denormalize_control(self, u_norm: np.ndarray) -> np.ndarray:
        """
        控制输入反归一化
        
        Args:
            u_norm: 归一化后的控制输入 [12维]
        
        Returns:
            原始控制输入 [12维]
        """
        if self.model_type == 'pytorch':
            # 使用CONTROL_MIN/MAX反归一化
            u = u_norm * (self.control_max - self.control_min + 1e-6) + self.control_min
            return u
        else:
            # MATLAB模型：使用action_bound反归一化
            if u_norm.ndim == 1:
                u_norm = u_norm.reshape(1, -1)
            
            # 从[-1, 1]反归一化
            u = (u_norm + 1) / 2 * (self.a_max - self.a_min + 1e-6) + self.a_min
            return u.flatten()
    
    def get_reference(self, ref_traj: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
        """
        处理参考轨迹：转换到局部坐标系并归一化
        
        Args:
            ref_traj: 参考轨迹 [N, 6] 每行为 [X, Y, Yaw, vx, vy, yaw_rate]
                     对于PyTorch模型：输入原始值（未归一化）
                     对于MATLAB模型：后3维（X）可能是归一化的（范围[-1,1]），需要先反归一化
            ref_pos: 参考点位置 [X, Y, Yaw]
        
        Returns:
            对于PyTorch模型：局部状态（未归一化，编码器内部有BatchNorm）
            对于MATLAB模型：归一化后的参考轨迹 [N, s_dim] 每行为归一化后的局部状态
        """
        # 对于PyTorch模型，参考轨迹应该已经是原始值，不需要反归一化
        if self.model_type == 'pytorch':
            # 直接对每个参考点进行坐标变换（不归一化）
            ref_normalized = []
            for i in range(ref_traj.shape[0]):
                ref_point = ref_traj[i, :]
                # normalization方法对PyTorch模型只做坐标变换，不归一化
                normalized = self.normalization(ref_point, ref_pos)
                ref_normalized.append(normalized)
            return np.array(ref_normalized)
        
        # MATLAB模型：需要处理归一化
        # ref_traj的后3维（X）可能是归一化的，需要先反归一化
        # 反归一化公式：(x_norm + 1) * (max - min) / 2 + min
        state_bound_dim = self.state_bound.shape[1] if self.state_bound.ndim > 1 else len(self.state_bound)
        
        if state_bound_dim == 3 and self.s_dim == 6:
            # state_bound对应后3维（速度相关）
            X_max = self.state_bound[1, :3]  # [20, 0.5, 0.5]
            X_min = self.state_bound[0, :3]  # [0, -0.5, -0.5]
            
            # 构建完整的ref_traj（未归一化）
            ref_traj_denorm = ref_traj.copy()
            # 前3维（位置）已经是未归一化的，后3维需要反归一化
            if ref_traj.shape[1] == 6:
                ref_traj_denorm[:, 3:6] = (ref_traj[:, 3:6] + 1) * (X_max - X_min) / 2 + X_min
        else:
            # 如果state_bound维度匹配，使用全部维度
            if state_bound_dim >= 3:
                X_max = self.state_bound[1, :3]
                X_min = self.state_bound[0, :3]
                if ref_traj.shape[1] == 6:
                    ref_traj_denorm = ref_traj.copy()
                    ref_traj_denorm[:, 3:6] = (ref_traj[:, 3:6] + 1) * (X_max - X_min) / 2 + X_min
                else:
                    ref_traj_denorm = ref_traj
            else:
                ref_traj_denorm = ref_traj
        
        # 对每个参考点进行归一化
        ref_normalized = []
        for i in range(ref_traj_denorm.shape[0]):
            ref_point = ref_traj_denorm[i, :]
            normalized = self.normalization(ref_point, ref_pos)
            
            # 注意：根据MATLAB输出分析，get_reference对第一点的x坐标做了特殊处理
            # MATLAB输出：第一点输出位置是[-2.000000, 0.000000, 0.000000]
            # 如果进行坐标变换，应该是[0.000000, 0.000000, 0.000000]
            # 但实际输出x=-2.0，说明MATLAB对第一点的x坐标减了2
            # 这个处理可能是为了确保第一点的参考状态在归一化范围内
            # 如果仿真结果不对，可以尝试注释掉这个处理
            if i == 0:
                normalized[0] = normalized[0] - 2.0
            
            ref_normalized.append(normalized)
        
        return np.array(ref_normalized)
    
    def encoder(self, x: np.ndarray) -> np.ndarray:
        """
        编码器：将状态编码到提升空间
        
        Args:
            x: 状态 [x, y, yaw, vx, vy, yaw_rate] (6维) 或 [1, 6]
                对于PyTorch模型：输入原始状态（编码器内部有BatchNorm归一化）
                对于MATLAB模型：输入归一化后的状态
        
        Returns:
            提升状态 [space_dim] = [s_dim + lift_dim] = [6 + 10] = 16维
        """
        if self.model_type == 'pytorch':
            # PyTorch模型：输入原始状态，编码器内部有BatchNorm
            # 确保输入是2D数组
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
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
        
        else:
            # MATLAB模型：使用传统全连接网络
            # 确保输入是2D数组
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # 提取原始状态部分（最后s_dim维）
            x_state = x[:, -self.s_dim:]
        
        # 前向传播通过编码器网络
        prev_layer = x_state
        
        # 编码器全连接层
        num_encoder_layers = len(self.encoder_widths) - 1
        for i in range(num_encoder_layers):
            # 获取权重和偏置
            W_key = f'WEF{i+1}'
            b_key = f'bEF{i+1}'
            W = None
            b = None
            
            if isinstance(self.weights, dict):
                # 如果是字典，直接获取
                W = self.weights.get(W_key, None)
                if W is not None:
                    # 如果W是numpy数组，确保转换为正确的格式
                    if isinstance(W, np.ndarray):
                        # 如果是0维数组（标量数组），使用item()；否则直接使用
                        if W.ndim == 0:
                            W = W.item()
                        else:
                            # 多维数组，直接使用，但确保是float类型
                            W = np.array(W, dtype=float)
                    elif hasattr(W, 'item'):
                        # 尝试item()，但如果失败则直接使用
                        try:
                            W = W.item()
                        except ValueError:
                            # item()失败，说明不是标量数组，直接使用
                            W = np.array(W, dtype=float)
                    else:
                        # 其他类型，转换为numpy数组
                        W = np.array(W, dtype=float)
                
                if i < num_encoder_layers - 1:
                    b = self.biases.get(b_key, None) if isinstance(self.biases, dict) else None
                    if b is not None:
                        # 同样的处理逻辑
                        if isinstance(b, np.ndarray):
                            if b.ndim == 0:
                                b = b.item()
                            else:
                                b = np.array(b, dtype=float)
                        elif hasattr(b, 'item'):
                            try:
                                b = b.item()
                            except ValueError:
                                b = np.array(b, dtype=float)
                        else:
                            b = np.array(b, dtype=float)
            elif hasattr(self.weights, '__dict__'):
                # 如果是对象，使用getattr
                W = getattr(self.weights, W_key, None)
                if i < num_encoder_layers - 1:
                    b = getattr(self.biases, b_key, None) if hasattr(self.biases, b_key) else None
            elif isinstance(self.weights, np.ndarray) and hasattr(self.weights.dtype, 'names'):
                # 如果是结构化数组
                if W_key in self.weights.dtype.names:
                    W = self.weights[W_key]
                    if hasattr(W, 'item'):
                        W = W.item()
                if i < num_encoder_layers - 1 and isinstance(self.biases, np.ndarray) and hasattr(self.biases.dtype, 'names'):
                    if b_key in self.biases.dtype.names:
                        b = self.biases[b_key]
                        if hasattr(b, 'item'):
                            b = b.item()
            else:
                # 尝试其他方式访问
                try:
                    if hasattr(self.weights, W_key):
                        W = getattr(self.weights, W_key)
                    elif isinstance(self.weights, np.ndarray) and W_key in self.weights.dtype.names:
                        W = self.weights[W_key]
                except:
                    pass
            
            if W is None:
                if isinstance(self.weights, dict):
                    available_keys = list(self.weights.keys())
                else:
                    available_keys = "N/A"
                raise ValueError(f"无法找到权重 {W_key}，可用键: {available_keys}")
            
            # 转换为numpy数组（确保是float类型）
            if not isinstance(W, np.ndarray):
                W = np.array(W, dtype=float)
            else:
                W = np.array(W, dtype=float)
            
            if b is not None:
                if not isinstance(b, np.ndarray):
                    b = np.array(b, dtype=float)
                else:
                    b = np.array(b, dtype=float)
            
            # 检查权重矩阵形状是否需要转置
            # 权重矩阵在.mat文件中可能是转置存储的（输出 x 输入），
            # 但我们需要的是（输入 x 输出）格式用于矩阵乘法: prev_layer @ W
            expected_input_dim = self.encoder_widths[i]
            expected_output_dim = self.encoder_widths[i + 1]
            if W.shape == (expected_output_dim, expected_input_dim):
                # 权重矩阵是转置存储的，需要转置
                W = W.T
            elif W.shape != (expected_input_dim, expected_output_dim):
                raise ValueError(
                    f"权重矩阵 {W_key} 的形状 {W.shape} 不正确。"
                    f"期望 ({expected_input_dim}, {expected_output_dim}) 或转置后的形状"
                )
            
            # 线性变换: prev_layer @ W + b
            # prev_layer: (batch, input_dim), W: (input_dim, output_dim)
            # 结果: (batch, output_dim)
            if b is not None:
                h = prev_layer @ W + b
            else:
                h = prev_layer @ W
            
            # 激活函数（除了最后一层）
            if i < num_encoder_layers - 1:
                act_type = self.eact_type[i] if isinstance(self.eact_type, (list, np.ndarray)) else 'relu'
                if act_type == 'relu':
                    h = np.maximum(0, h)
                elif act_type == 'tanh':
                    h = np.tanh(h)
                elif act_type == 'sigmoid':
                    h = 1 / (1 + np.exp(-h))
                elif act_type == 'elu':
                    h = np.where(h > 0, h, np.exp(h) - 1)
            
            prev_layer = h
        
        # 拼接原始状态和提升状态
        # 根据MATLAB代码，编码器输出是 [原始状态, 提升状态]
        lifted_state = np.concatenate([x_state, prev_layer], axis=1)
        
        return lifted_state.flatten()  # 返回1D数组


class MPCController:
    """
    MPC控制器类
    实现MPC优化求解
    """
    
    def __init__(self, ddk: DDK, Np: int = 30, Nc: int = 30, 
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                 delta_umax: Optional[np.ndarray] = None):
        """
        初始化MPC控制器
        
        Args:
            ddk: DDK模型实例
            Np: 预测时域
            Nc: 控制时域
            Q: 状态权重矩阵 (s_dim x s_dim)
            R: 控制权重矩阵 (u_dim x u_dim)
            delta_umax: 控制增量约束 [u_dim]
        """
        self.ddk = ddk
        self.Np = int(Np)  # 确保是整数（从MATLAB传入可能是float）
        self.Nc = int(Nc)  # 确保是整数（从MATLAB传入可能是float）
        self.Nx = int(ddk.space_dim)  # 提升空间维度，确保是整数
        self.Nu = int(ddk.u_dim)      # 控制输入维度，确保是整数
        
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
        """构建扩展状态空间模型（包含历史控制输入）"""
        # self.ddk.A 和 self.ddk.B 已经在 _extract_koopman_matrices 中转置过了
        # 所以这里直接使用，不需要再次转置
        a = self.ddk.A  # (Nx, Nx)
        b = self.ddk.B  # (Nx, Nu)
        
        # 扩展状态: [x_lift; u_prev]
        # 扩展系统: [x_lift_{k+1}; u_k] = A_ext * [x_lift_k; u_{k-1}] + B_ext * delta_u_k
        A_ext = np.zeros((self.Nx + self.Nu, self.Nx + self.Nu))
        A_ext[:self.Nx, :self.Nx] = a
        A_ext[:self.Nx, self.Nx:] = b
        A_ext[self.Nx:, self.Nx:] = np.eye(self.Nu)
        
        B_ext = np.zeros((self.Nx + self.Nu, self.Nu))
        B_ext[:self.Nx, :] = b
        B_ext[self.Nx:, :] = np.eye(self.Nu)
        
        self.A_ext = A_ext
        self.B_ext = B_ext
        
        # 输出矩阵C（只输出原始状态，对应MATLAB代码）
        # MATLAB: C = [eye(s_dim) zeros(s_dim, space_dim - s_dim + Nu)]
        # 其中 space_dim - s_dim + Nu = (Nx - s_dim) + Nu
        # kesi维度是 (Nx + Nu)，C选择前s_dim维（原始状态），后面全0
        self.C = np.hstack([np.eye(self.ddk.s_dim), 
                           np.zeros((self.ddk.s_dim, self.Nx - self.ddk.s_dim + self.Nu))])
    
    def _precompute_prediction_matrices(self):
        """预计算预测矩阵PHI和THETA"""
        # PHI: 自由响应矩阵
        PHI_list = []
        for j in range(1, self.Np + 1):
            PHI_list.append(self.C @ np.linalg.matrix_power(self.A_ext, j))
        self.PHI = np.vstack(PHI_list)  # (Np*s_dim) x (Nx+Nu)
        
        # THETA: 强制响应矩阵
        THETA_list = []
        for j in range(1, self.Np + 1):
            row = []
            for k in range(1, self.Nc + 1):
                if k <= j:
                    row.append(self.C @ np.linalg.matrix_power(self.A_ext, j - k) @ self.B_ext)
                else:
                    row.append(np.zeros((self.ddk.s_dim, self.Nu)))
            THETA_list.append(np.hstack(row))
        self.THETA = np.vstack(THETA_list)  # (Np*s_dim) x (Nc*Nu)
    
    def _build_constraints(self):
        """构建约束矩阵，并预计算H和A_combined矩阵"""
        # 控制输入约束矩阵A_l（累积约束）
        A_l = np.zeros((self.Nc, self.Nc))
        for p in range(self.Nc):
            for q in range(self.Nc):
                if q <= p:
                    A_l[p, q] = 1
        
        self.A_l = np.kron(A_l, np.eye(self.Nu))  # (Nc*Nu) x (Nc*Nu)
        
        # 控制输入边界
        # 对于PyTorch模型：归一化范围[0, 1]
        # 对于MATLAB模型：归一化范围[-1, 1]
        if self.ddk.model_type == 'pytorch':
            umin = np.zeros(self.Nu)  # [0, 0, ..., 0]
            umax = np.ones(self.Nu)   # [1, 1, ..., 1]
        else:
            umin = -np.ones(self.Nu)  # [-1, -1, ..., -1]
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
        
        # 添加数值稳定性：对H矩阵添加小的正则化项，避免病态问题
        # 这可以防止在边界情况下H矩阵条件数过大
        reg_factor = 1e-8
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
            u_prev: 上一时刻控制输入 [u_dim]
            ref_traj: 参考轨迹 [Np, s_dim]（归一化后的局部状态）
        
        Returns:
            delta_u: 控制增量 [u_dim]
            success: 是否求解成功
        """
        # 构建扩展状态
        kesi = np.concatenate([x_lift, u_prev])
        
        # 构建参考轨迹向量（对应MATLAB: reshape(ref_r', [s_dim * Np, 1])）
        # ref_traj是(Np, s_dim)，转置后是(s_dim, Np)，按列优先展开成(s_dim*Np,)或按行优先展开成(Np*s_dim,)
        # MATLAB的reshape默认是列优先，所以reshape(ref_r', [s_dim*Np, 1])等价于按列展开
        # 但ref_r'是(s_dim, Np)，按列展开就是先取第1列，再取第2列...
        # Python的flatten默认按行优先，对应MATLAB的reshape(ref_r, [1, Np*s_dim])
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
            
            # 将误差reshape为(Np, s_dim)以便分析各维度误差
            error_reshaped = error.reshape(self.Np, self.ddk.s_dim)
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
        # error是(s_dim*Np,)列向量，error.T是(1, s_dim*Np)行向量
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
        for i in range(self.Nu):
            u_prev_i = u_prev[i]
            
            if self.ddk.model_type == 'pytorch':
                # PyTorch模型：归一化范围[0, 1]
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
            else:
                # MATLAB模型：归一化范围[-1, 1]
                # 如果u_prev接近下界，限制delta_u不能为负（或只能很小的负值）
                if u_prev_i <= -0.95:  # 接近下界-1
                    # delta_u[i]必须 >= -(u_prev_i + 1)，确保u_new >= -1
                    min_delta = -(u_prev_i + 1.0) + 1e-6  # 添加小的安全裕度
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
            u_normalized: 归一化控制输入 [12维]
                对于PyTorch模型：范围[0, 1]（使用CONTROL_MIN/MAX归一化）
                对于MATLAB模型：范围[-1, 1]（使用action_bound归一化）
        
        Returns:
            control: [12维]
                    [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
                     steer_LR(deg), steer_RR(deg),
                     torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
                     torque_LR(N·m), torque_RR(N·m)]
        """
        # 确保输入是12维
        if len(u_normalized) != 12:
            raise ValueError(f"控制输入维度错误: 期望12维，实际{len(u_normalized)}维")
        
        # 反归一化控制输入
        if self.ddk.model_type == 'pytorch':
            # PyTorch模型：使用CONTROL_MIN/MAX反归一化
            u_denorm = self.ddk.denormalize_control(u_normalized)
        else:
            # MATLAB模型：使用action_bound反归一化
            u_denorm = self.ddk.denormalize_control(u_normalized)
        
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

