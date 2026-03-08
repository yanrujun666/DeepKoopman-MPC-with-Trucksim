import os
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import scipy.io as scio
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor

from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset


#######################

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


class BatchFourierPositionalEncoding(nn.Module):
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
        coords_normalized = coords
        angle = coords_normalized.unsqueeze(-1) * self.freqs.view(1, 1, 1, 1, -1)
        sin_enc = torch.sin(2 * torch.pi * angle)  # (..., input_dim, L)
        cos_enc = torch.cos(2 * torch.pi * angle)  # (..., input_dim, L)
        
        # 将正弦和余弦交错排列
        pe = torch.stack([sin_enc, cos_enc], dim=-1)  # (..., input_dim, L, 2)
        pe = pe.flatten(start_dim=-3, end_dim=-1)     # (..., input_dim * L * 2)
        pe = torch.cat([coords, pe], dim=-1)
        
        return pe
    

class CustomEncoderUnscaledv2WithoutNorm(nn.Module):

    def __init__(self, state_dim, hidden_dim, layer_depth):
        super().__init__()

        self.in_embed = BatchFourierPositionalEncoding(L=8, input_dim=1)

        self.channel_fc = nn.Sequential(
            nn.Linear(16+1, 64),
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

        self.skip_fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim-state_dim),
        )
        
    def forward(self, x):

        B, S, C = x.shape        
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

        skip_embed = self.skip_fc(x)

        return self.out_fc(fused_embed) + skip_embed


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

def load_vehicle_data(mat_path):
    """
    :param mat_path: MAT文件路径
    :return: 结构化字典数据，包含actuators、vehicle_dynamics和initial_conditions
    """
    try:
        # 加载原始数据（自动转换struct为字典）
        raw_data = scio.loadmat(mat_path, simplify_cells=True)
        
        # 数据架构容器
        parsed_data = {
            'actuators': {},
            'vehicle_dynamics': {},
            'initial_conditions': {}  # 新增初始条件容器
        }

        #-------------------------- 初始条件提取 --------------------------#
        # 检查是否存在initial_conditions字段
        if 'initial_conditions' in raw_data:
            # MATLAB结构体可能被转换为字典或numpy.void对象
            # 这里假设已经是简化后的字典结构
            initial_cond = raw_data['initial_conditions']
            
            # 提取目标变量（注意MATLAB变量名大小写敏感性）
            parsed_data['initial_conditions'] = {
                'wz0': initial_cond.get('wz0', None),  # 初始横摆角速度
                'vx0': initial_cond.get('vx0', None),  # 初始纵向速度
                'vy0': initial_cond.get('vy0', None),  # 初始横向速度
                'w0': initial_cond.get('w0', None)      # 初始车轮转速（可能需要确认命名）
            }
        else:
            print("警告: 未找到initial_conditions字段")

        if len(raw_data['Delta_act']) != len(raw_data['T_motor']):
            length = len(raw_data['T_motor'])
            raw_data['Delta_act'] = np.expand_dims(raw_data['Delta_act'], 0).repeat(length, 0)
        
        assert len(raw_data['Delta_act']) == len(raw_data['T_motor'])

        #-------------------------- Actuator Data --------------------------#
        parsed_data['actuators']['steer_angles'] = {
            k: raw_data['Delta_act'][:, i].astype(np.float32)
            for i, k in enumerate([
                'FL_steer', 'FR_steer',
                'ML_steer', 'MR_steer',
                'RL_steer', 'RR_steer'
            ])
        }

        parsed_data['actuators']['wheel_torques'] = {
            k: raw_data['T_motor'][:, i].astype(np.float32)
            for i, k in enumerate([
                'FL_trq', 'FR_trq',
                'ML_trq', 'MR_trq',
                'RL_trq', 'RR_trq'
            ])
        }

        #-------------------------- Vehicle Dynamics --------------------------#
        state_labels = [
            'Xpos', 'Ypos', 'Yaw',          # 0-2
            'Vx', 'Vy', 'YawRate',         # 3-5
            'Ax', 'Ay', 'YawAccel',         # 6-8
            *[f'WhlSpd_{pos}' for pos in ['FL', 'FR', 'ML', 'MR', 'RL', 'RR']],  # 9-14
            *[f'{force_direction}_{pos}' 
              for force_direction in ['Fx', 'Fy', 'Fz']
              for pos in ['FL', 'FR', 'ML', 'MR', 'RL', 'RR']
            ],  # 15-32
            'Alpha_FL', 'Alpha_FR', 'Alpha_ML', 'Alpha_MR', 'Alpha_RL', 'Alpha_RR',
            'Vy_processed'
        ]

        parsed_data['vehicle_dynamics'] = {
            k: raw_data['Vehicle_state_trucksim_39d'][:, i].astype(np.float32)
            for i, k in enumerate(state_labels[:39])
        }

        return parsed_data

    except Exception as e:
        print(f"数据解析失败: {str(e)}, {str(mat_path)}")
        print(raw_data)
        exit(0)


def preprocess_trucksim_data_7dof(parsed_data):
    # 获取原始数据引用
    actuators = parsed_data['actuators']
    
    # 转角和扭矩按轴分组索引
    axis_indices = [
        ['FL'], 
        ['FR'],
        ['ML'],
        ['MR'], 
        ['RL'], 
        ['RR'] 
    ]
    
    # 计算各轴平均转角和总扭矩
    delta_axis = []
    torque_axis = []
    for idx in axis_indices:
        # 转角
        delta_avg = actuators['steer_angles'][idx[0]+'_steer']
        delta_axis.append(delta_avg)
        
        # 扭矩
        torque_sum = actuators['wheel_torques'][idx[0]+'_trq']
        torque_axis.append(torque_sum)
    
    # 存储处理后的控制输入
    parsed_data['control_input'] = {
        'delta': np.column_stack(delta_axis),  # 各轴转角 [前,中,后]
        'torque': np.column_stack(torque_axis) # 各轴扭矩 [前,中,后]
    }
    
    return parsed_data


class VehicleDynamicDataset5C(Dataset):
    
    def __init__(self, mat_fpath, horizon=100, device=torch.device('cpu'), show_log=True, mode='7dof', skip=100):
        super(VehicleDynamicDataset5C, self).__init__()
        # load trucksim data
        self.trucksim_data_raw = load_vehicle_data(mat_fpath)
        self.mode = mode.lower()
        assert self.mode in ['3dof', '7dof']
        # preprocess
        self.trucksim_data_ppd = preprocess_trucksim_data_7dof(self.trucksim_data_raw)
        
        # show log
        if show_log:
            for k in self.trucksim_data_ppd.keys():
                tmp = self.trucksim_data_ppd[k]
                if isinstance(tmp, dict):
                    for kk in tmp.keys():
                        if isinstance(tmp[kk], dict):
                            for kkk in tmp[kk].keys():
                                print('%s [%s] (%s): ' % (k, kk, kkk), np.array(tmp[kk][kkk]).shape)
                        else:
                            print('%s [%s]: ' % (k, kk), np.array(tmp[kk]).shape)
                else:
                    print('%s: ' % k, np.array(tmp).shape)

        condition = mat_fpath.split('/')[1].split('_')[0]
        condition_embed = torch.zeros(5)
        if condition == 'c1':
            condition_embed[0] = 1
        elif condition == 'c2':
            condition_embed[1] = 1
        elif condition == 'c3':
            condition_embed[2] = 1
        elif condition == 'c4':
            condition_embed[3] = 1
        elif condition == 'c5':
            condition_embed[4] = 1
        elif condition == 'all':
            condition = mat_fpath.split('/')[-1].split('_')[0]
            if condition == 'all':
                condition_embed[0] = 1
            elif condition == 'counter':
                condition_embed[1] = 1
            elif condition == 'arckman':
                condition_embed[2] = 1
            elif condition == 'lateral':
                condition_embed[3] = 1
            elif condition == 'crab':
                condition_embed[4] = 1
            elif condition == 'rollover':
                condition_embed[0] = 1
            else:
                IndexError('unknown condition!')
        else:
            raise IndexError('unknown condition!')

        self.condition = condition_embed

        self.device = device

        self.skip = skip

        if horizon == -1:
            self.horizon = len(self.trucksim_data_ppd['control_input']['delta']) - self.skip - 1
        else:
            self.horizon = horizon
        
        self.total_samples = len(self.trucksim_data_ppd['control_input']['delta']) - self.horizon - self.skip
        # print(self.total_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        current_idx = idx + self.skip
        future_idx = idx + self.horizon +self.skip

        # get a clip of data
        clip = {

            # 控制输入
            'delta_act': {
                'total': torch.from_numpy(self.trucksim_data_ppd['control_input']['delta'][current_idx:future_idx, :]).to(self.device), # t ~ t+N
                'fl': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['FL_steer'][current_idx:future_idx]).to(self.device),
                'fr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['FR_steer'][current_idx:future_idx]).to(self.device),
                'ml': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['ML_steer'][current_idx:future_idx]).to(self.device),
                'mr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['MR_steer'][current_idx:future_idx]).to(self.device),
                'rl': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['RL_steer'][current_idx:future_idx]).to(self.device),
                'rr': torch.from_numpy(self.trucksim_data_ppd['actuators']['steer_angles']['RR_steer'][current_idx:future_idx]).to(self.device),
            },
            'T_motor': {
                'total': torch.from_numpy(self.trucksim_data_ppd['control_input']['torque'][current_idx:future_idx, :]).to(self.device), # t ~ t+N
                'fl': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['FL_trq'][current_idx:future_idx]).to(self.device),
                'fr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['FR_trq'][current_idx:future_idx]).to(self.device),
                'ml': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['ML_trq'][current_idx:future_idx]).to(self.device),
                'mr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['MR_trq'][current_idx:future_idx]).to(self.device),
                'rl': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['RL_trq'][current_idx:future_idx]).to(self.device),
                'rr': torch.from_numpy(self.trucksim_data_ppd['actuators']['wheel_torques']['RR_trq'][current_idx:future_idx]).to(self.device),
            },

            # 状态序列
            'vx_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Vx'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, vx
            'vy_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Vy'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, vy
            'wz_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['YawRate'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, wz
            'x_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Xpos'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, x
            'y_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ypos'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, y
            'thetaz_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Yaw'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, z
            'ax_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ax'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, ax
            'ay_est_gt': torch.from_numpy(self.trucksim_data_ppd['vehicle_dynamics']['Ay'][current_idx:future_idx+1]).to(self.device), # t ~ t+N+1, ay
        
            # 时间戳
            'his_timestamp': torch.from_numpy(np.arange(current_idx, future_idx)), # t ~ t+N
            'est_timestamp': torch.tensor(future_idx), # t+N+1

            'condition': self.condition.to(self.device)
        }

        assert clip['delta_act']['total'].shape[0] == self.horizon
        assert clip['vx_est_gt'].shape[0] == self.horizon + 1

        return clip
    
#######################
# default
condition_dir_dict = {
    'all_wheel_steer': 'data/c1_all_wheel_steer/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA',
    'counter_rotation': 'data/c2_counter_rot/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA',
    'arckman': 'data/c3_arckman/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA',
    'lateral_movement': 'data/c4_steer90/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA',
    'crab_movement': 'data/c5_crab/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA',
    'rollover': 'data/c1_rollover/test/',
    'all': 'data/all',
    'all_rollover': 'data/all_rollover'
}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

def eval(encoder, dkm, dst):
    qbar = tqdm.tqdm(dst)
    encoder.eval()
    dkm.eval()

    dkm_avg_eval_error = 0.0
    dkm_final_eval_error = 0.0
    yaw_avg_eval_error = 0.0
    yaw_final_eval_error = 0.0
    vx_eval_error = 0.0
    vy_eval_error = 0.0
    wz_eval_error = 0.0
    sample_cnt = 0

    for i, sample in enumerate(qbar):
        with torch.no_grad():
            states = torch.stack([
                sample['x_est_gt'],
                sample['y_est_gt'],
                sample['thetaz_est_gt'],
                sample['vx_est_gt'],
                sample['vy_est_gt'],
                sample['wz_est_gt'],
            ], dim=-1)
            controls = torch.cat([
                sample['T_motor']['total'],
                sample['delta_act']['total']
            ], dim=-1)
            states[:, :, :2] -= states[:, 0:1, :2].repeat(1, states.shape[1], 1)
            controls_norm = (controls - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)
            state_embeds_res = encoder(states)
            state_embeds_full = torch.cat([states, state_embeds_res], dim=-1)
            ITERS = states.shape[1] - 1
            result_embeds_full = dkm(state_embeds_full[:, :-1, :], controls_norm, iters=ITERS)
            est_states = result_embeds_full[..., :6]

            # save_figname = os.path.join('viz', '%06d.png' % i)
            # viz(states, est_states, save_figname)
        
        dkm_loss = torch.norm(est_states[:, :, :2] - states[:, 1:, :2], p=2, dim=-1) # multi-step prediction loss in state space
        yaw_loss = torch.abs(est_states[:, :, 2] - states[:, 1:, 2])
        
        dkm_avg_eval_error += torch.sum(dkm_loss)
        yaw_avg_eval_error += torch.sum(yaw_loss)

        fde_dkm_loss = torch.norm(est_states[:, -1, :2] - states[:, -1, :2],  p=2, dim=-1)
        yaw_dkm_loss = torch.abs(est_states[:, -1, 2] - states[:, -1, 2])

        dkm_final_eval_error += torch.sum(fde_dkm_loss)
        yaw_final_eval_error += torch.sum(yaw_dkm_loss)

        vx_eval_error += torch.sum(torch.abs(est_states[:, :, 3] - states[:, 1:, 3]))
        vy_eval_error += torch.sum(torch.abs(est_states[:, :, 4] - states[:, 1:, 4]))
        wz_eval_error += torch.sum(torch.abs(est_states[:, :, 5] - states[:, 1:, 5]))
        
        sample_cnt += states.shape[0]
    
    dkm_avg_eval_error /= (sample_cnt * ITERS)
    dkm_final_eval_error /= sample_cnt
    yaw_avg_eval_error /= (sample_cnt * ITERS)
    yaw_avg_eval_error *= (180.0 / np.pi)
    yaw_final_eval_error /= sample_cnt
    yaw_final_eval_error *= (180.0 / np.pi)
    vx_eval_error /= (sample_cnt * ITERS)
    vy_eval_error /= (sample_cnt * ITERS)
    wz_eval_error /= (sample_cnt * ITERS)
    
    print('Valid:')
    print('DKM model ADE: %.4f' % dkm_avg_eval_error.item(), 'FDE: %.4f' % dkm_final_eval_error.item())
    print('DKM model average yaw error: %.4f' % yaw_avg_eval_error.item(), 'final yaw error: %.4f' % yaw_final_eval_error.item())
    print('vx_eval_error: %.4f' % vx_eval_error.item(), 'vy_eval_error: %.4f' % vy_eval_error.item(), 'wz_eval_error: %.4f' % (wz_eval_error.item() * (180.0/np.pi)))

    return dkm_avg_eval_error, dkm_final_eval_error, vx_eval_error, vy_eval_error, wz_eval_error, yaw_avg_eval_error, yaw_final_eval_error


def viz(states, est_states, figname):
    states_npy = states[0].cpu().numpy()
    est_states_npy = np.concatenate([states_npy[:1], est_states[0].cpu().numpy()], axis=0)
            
    assert len(est_states_npy) == len(states_npy)
    timestamps = np.arange(len(est_states_npy)) * 0.01
            
    # visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    fig.suptitle('visualization of DeepKoopman', fontsize=16, fontweight='bold')
            
    axes_flat = axes.flatten()

    axes_flat[0].plot(timestamps, est_states_npy[:, 0], 'g-o', linewidth=2, label='dkm')
    axes_flat[0].plot(timestamps, states_npy[:, 0], 'b-', linewidth=2, label='gt')
    axes_flat[0].set_title(f'X', fontsize=12, fontweight='bold')
    axes_flat[0].set_xlabel('timestamp')
    axes_flat[0].set_ylabel('x')
    axes_flat[0].grid(True, alpha=0.3)
    axes_flat[0].legend(loc='best')

    axes_flat[1].plot(timestamps, est_states_npy[:, 1], 'g-o', linewidth=2, label='dkm')
    axes_flat[1].plot(timestamps, states_npy[:, 1], 'b-', linewidth=2, label='gt')
    axes_flat[1].set_title(f'Y', fontsize=12, fontweight='bold')
    axes_flat[1].set_xlabel('timestamp')
    axes_flat[1].set_ylabel('y')
    axes_flat[1].grid(True, alpha=0.3)
    axes_flat[1].legend(loc='best')

    axes_flat[2].plot(timestamps, est_states_npy[:, 2], 'g-o', linewidth=2, label='dkm')
    axes_flat[2].plot(timestamps, states_npy[:, 2], 'b-', linewidth=2, label='gt')
    axes_flat[2].set_title(f'Yaw', fontsize=12, fontweight='bold')
    axes_flat[2].set_xlabel('timestamp')
    axes_flat[2].set_ylabel('yaw')
    axes_flat[2].grid(True, alpha=0.3)
    axes_flat[2].legend(loc='best')

    axes_flat[3].plot(est_states_npy[:, 0], est_states_npy[:, 1], 'g-o', linewidth=2, label='dkm')
    axes_flat[3].plot(states_npy[:, 0], states_npy[:, 1], 'b-', linewidth=2, label='gt')
    axes_flat[3].set_title(f'trajectory', fontsize=12, fontweight='bold')
    axes_flat[3].set_xlabel('x')
    axes_flat[3].set_ylabel('y')
    axes_flat[3].grid(True, alpha=0.3)
    axes_flat[3].legend(loc='best')

    axes_flat[4].plot(timestamps, abs(est_states_npy[:, 3] - states_npy[:, 3]), 'g-o', linewidth=2, label='error')
    axes_flat[4].set_title(f'X error', fontsize=12, fontweight='bold')
    axes_flat[4].set_xlabel('timestamp')
    axes_flat[4].set_ylabel(f'e_x')
    axes_flat[4].grid(True, alpha=0.3)
    axes_flat[4].legend(loc='best')

    axes_flat[5].plot(timestamps, abs(est_states_npy[:, 4] - states_npy[:, 4]), 'g-o', linewidth=2, label='error')
    axes_flat[5].set_title(f'Y error', fontsize=12, fontweight='bold')
    axes_flat[5].set_xlabel('timestamp')
    axes_flat[5].set_ylabel(f'e_y')
    axes_flat[5].grid(True, alpha=0.3)
    axes_flat[5].legend(loc='best')

    axes_flat[6].plot(timestamps, abs(est_states_npy[:, 5] - states_npy[:, 5]), 'g-o', linewidth=2, label='error')
    axes_flat[6].set_title(f'Yaw error', fontsize=12, fontweight='bold')
    axes_flat[6].set_xlabel('timestamp')
    axes_flat[6].set_ylabel(f'e_yaw')
    axes_flat[6].grid(True, alpha=0.3)
    axes_flat[6].legend(loc='best')
    
    error_traj = np.sqrt(abs(est_states_npy[:, 0] - states_npy[:, 0])**2 + abs(est_states_npy[:, 1] - states_npy[:, 1])**2)
    axes_flat[7].plot(timestamps, error_traj, 'g-o', linewidth=2, label='error')
    axes_flat[7].set_title(f'Trajectory error', fontsize=12, fontweight='bold')
    axes_flat[7].set_xlabel('timestamp')
    axes_flat[7].set_ylabel(f'e_traj')
    axes_flat[7].grid(True, alpha=0.3)
    axes_flat[7].legend(loc='best')

    plt.tight_layout()

    plt.savefig(figname)
    plt.close()


if __name__ == '__main__':
    #######################
    # choose eval condition
    eval_condition = 'all_rollover'
    # eval_condition = 'all'
    eval_condition = eval_condition.lower()
    assert eval_condition in list(condition_dir_dict.keys()), 'no such condition'

    #######################
    # choose eval length
    eval_length = 100
    if eval_length >= 300:
        skip = 25
    else:
        skip = 100
    valid_dir = condition_dir_dict[eval_condition]
    valid_files = [os.path.join(valid_dir, x) for x in sorted(os.listdir(valid_dir))]
    valid_dsts = []
    for file in valid_files:
        valid_dsts.append(VehicleDynamicDataset5C(mat_fpath=file, horizon=eval_length, device='cuda:0', show_log=False, skip=skip))
    valid_dst = ConcatDataset(valid_dsts)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=512, shuffle=False, num_workers=0, drop_last=False)

    #######################
    # config model
    def build_model(ckpt_path):
        encoder = CustomEncoderUnscaledv2WithoutNorm(state_dim=6, hidden_dim=16, layer_depth=6)
        dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)

        print('Params of encoder:', count_parameters(encoder) / 1e6, 'M')
        print('Params of dkm:', count_parameters(dkm) / 1e6, 'M')

        ckpt = torch.load(ckpt_path)
        # print(ckpt['koopman'])
        print(ckpt['eval_results'])
        print(ckpt['epoch'])
        encoder.load_state_dict(ckpt['encoder'])
        dkm.load_state_dict(ckpt['koopman'])

        encoder = encoder.to(DEVICE)
        dkm = dkm.to(DEVICE)
        return encoder, dkm

    encoder, dkm = build_model(
        ckpt_path='DeepEDMD-Transv2wonorm-hd16-multiset-100e-remote-local-lr1e-4-rollover-0.05pilossv24-0222.pth'
    )

    eval(encoder, dkm, valid_loader)