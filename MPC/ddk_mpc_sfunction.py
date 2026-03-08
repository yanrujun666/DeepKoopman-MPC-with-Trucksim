"""
Python MPC控制器接口模块
用于在Simulink中通过MATLAB Function模块调用

使用方法：
1. 在Simulink中添加"MATLAB Function"块
2. 使用matlab_function_wrapper.m包装此模块
3. 配置输入输出端口

注意：此模块通过MATLAB Function模块调用，不是直接的Python S-Function
"""

import os
import sys
from pathlib import Path

# 解决OpenMP冲突问题（必须在导入numpy之前设置）
# MATLAB和Python的科学计算库都使用了OpenMP，会导致冲突
# 设置此环境变量允许程序继续执行（虽然可能有轻微性能影响，但避免了崩溃）
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

# 确保模块目录在Python路径中（用于MATLAB调用）
_module_dir = os.path.dirname(os.path.abspath(__file__))
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

# 导入新 Koopman-MPC V2（方案 A：新网络 + 新 MPC，不依赖 mpc_dk）
try:
    from koopman_mpc_v2 import (
        load_model,
        KoopmanMPC,
        encode_state,
        encode_ref_trajectory,
        convert_to_control_output,
        CONTROL_MIN_NP,
        CONTROL_MAX_NP,
    )
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from koopman_mpc_v2 import (
        load_model,
        KoopmanMPC,
        encode_state,
        encode_ref_trajectory,
        convert_to_control_output,
        CONTROL_MIN_NP,
        CONTROL_MAX_NP,
    )


# 全局状态（在S-Function调用之间保持）
_controller_state = {
    'encoder': None,
    'dkm': None,
    'mpc': None,
    'device': None,
    'ref_traj': None,
    'u_prev': 0.5 * np.ones(12),
    'nearest_idx': 1,
    'start_idx': 1,
    'index': 0,
    'mpc_step': 0,
    'decimation': 10,
    'last_control_output': np.zeros(12).reshape(12, 1),
    'bad_count': 0,
    'last_state': None,
    'sample_interval': 1,  # 1=与 Koopman 0.01s 对齐，ref_traj_np[k] 对应 (nearest_idx+k)*0.01s
    'Np': 30,
    'u_init_prev': None,
    'u_last_output': None,   # 用于输出平滑的上一拍实际下发控制（归一化 [0,1]^12）
    'smooth_alpha': 0.45,    # 输出一阶滤波系数：u_out = alpha*u_last + (1-alpha)*u_mpc，越大越平滑
}


def find_nearest_point(ref_traj, current_state, start_idx, search_range):
    """查找参考轨迹中最近的点"""
    start_idx = int(start_idx)
    end_idx = int(min(start_idx + search_range, len(ref_traj)))
    search_traj = ref_traj[start_idx:end_idx, :]
    
    if len(search_traj) == 0:
        return start_idx, float('inf')
    
    distances = np.sum((search_traj[:, :3] - current_state[:3])**2, axis=1) + \
                0.1 * np.sum((search_traj[:, 3:6] - current_state[3:6])**2, axis=1)
    
    min_idx = np.argmin(distances)
    nearest_idx = int(start_idx + min_idx)
    min_distance = distances[min_idx]
    
    return nearest_idx, min_distance


def initialize_controller(param_path, data_path, Np=30, Nc=30, sample_interval=1, decimation=10):
    """
    初始化 MPC 控制器（方案 A：新网络 + 新 KoopmanMPC）
    Args:
        param_path: 新 ckpt 路径（.pth）
        data_path: 参考轨迹 .mat 路径
        Np: 预测时域（与 horizon 一致，默认 30）
        Nc: 未使用（保留接口兼容）
        sample_interval: 参考轨迹采样间隔（1=与 Koopman 0.01s 对齐，ref_traj_np[k] 与 z[k+1] 时间一致）
        decimation: 每 decimation 次调用求解一次 MPC（10 -> 10ms，与 new code 对齐）
    """
    global _controller_state
    import torch

    print(f"[Python S-Function] 初始化控制器 (Koopman-MPC V2)...")
    print(f"  参数文件: {param_path}")
    print(f"  数据文件: {data_path}")

    device = torch.device("cpu")
    _controller_state['device'] = device
    encoder, dkm = load_model(param_path, device=device)
    _controller_state['encoder'] = encoder
    _controller_state['dkm'] = dkm
    _controller_state['mpc'] = KoopmanMPC(
        dkm,
        state_dim=6,
        control_dim=12,
        horizon=int(Np),
        control_min_np=CONTROL_MIN_NP,
        control_max_np=CONTROL_MAX_NP,
        device=device,
    )

    # 仿真开始打印 A、B 矩阵统计
    mpc = _controller_state['mpc']
    A_np, B_np = mpc.A_np, mpc.B_np
    print("[MPC] Koopman 矩阵统计 (仿真开始):")
    print(f"  A: shape={A_np.shape}, norm={np.linalg.norm(A_np):.6f}, max={np.max(np.abs(A_np)):.6f}, cond={np.linalg.cond(A_np):.4f}")
    try:
        eigA = np.linalg.eigvals(A_np)
        print(f"  A 特征值: 实部=[{eigA.real.min():.4f}, {eigA.real.max():.4f}], 最大模={np.max(np.abs(eigA)):.4f}")
    except Exception:
        pass
    print(f"  B: shape={B_np.shape}, norm={np.linalg.norm(B_np):.6f}, max={np.max(np.abs(B_np)):.6f}")
    print(f"  B 列范数(按控制维): {np.linalg.norm(B_np, axis=0).round(6).tolist()}")

    # 加载参考数据
    # 支持三种数据格式：
    # 1. 新格式：ref_trajectory结构体（包含position和velocity子结构体）
    # 2. Trucksim格式：'position' 和 'velocity' 字段（N×3数组）
    # 3. Carsim格式：'Pos' 和 'X' 字段（N×3数组）
    import scipy.io as scio
    ref_data = scio.loadmat(data_path, squeeze_me=True)
    
    # 辅助函数：从MATLAB结构体中提取字段
    def extract_struct_field(struct_data, field_name):
        """从MATLAB结构体中提取字段值"""
        if isinstance(struct_data, np.ndarray) and struct_data.dtype.names:
            # MATLAB结构体数组（numpy结构化数组）
            if struct_data.shape == ():
                # 标量结构体
                if field_name in struct_data.dtype.names:
                    value = struct_data[field_name]
                    # 如果是0维数组，使用item()提取值
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        return value.item()
                    return value
                else:
                    raise KeyError(f"结构体中未找到字段'{field_name}'。可用字段: {struct_data.dtype.names}")
            else:
                # 数组结构体（通常不会出现，因为position和velocity是标量结构体）
                return np.array([struct_data[i][field_name] for i in range(len(struct_data))])
        elif hasattr(struct_data, field_name):
            # Python对象属性
            return getattr(struct_data, field_name)
        else:
            raise ValueError(f"无法从结构体中提取字段'{field_name}'，类型: {type(struct_data)}")
    
    # 检查是否是新格式（ref_trajectory结构体）
    if 'ref_trajectory' in ref_data:
        print(f"[Python S-Function] 检测到新格式参考轨迹（ref_trajectory结构体）")
        ref_traj_struct = ref_data['ref_trajectory']
        
        try:
            # 提取position结构体
            pos_struct = extract_struct_field(ref_traj_struct, 'position')
            
            # 从position结构体中提取x, y, yaw
            pos_x = extract_struct_field(pos_struct, 'x')
            pos_y = extract_struct_field(pos_struct, 'y')
            pos_yaw = extract_struct_field(pos_struct, 'yaw')
            
            # 提取velocity结构体
            vel_struct = extract_struct_field(ref_traj_struct, 'velocity')
            
            # 从velocity结构体中提取vx, vy, wz
            vel_vx = extract_struct_field(vel_struct, 'vx')
            vel_vy = extract_struct_field(vel_struct, 'vy')
            vel_wz = extract_struct_field(vel_struct, 'wz')
            
            # 确保所有数据都是numpy数组且形状一致
            pos_x = np.array(pos_x).flatten()
            pos_y = np.array(pos_y).flatten()
            pos_yaw = np.array(pos_yaw).flatten()
            vel_vx = np.array(vel_vx).flatten()
            vel_vy = np.array(vel_vy).flatten()
            vel_wz = np.array(vel_wz).flatten()
            
            # 检查长度一致性
            lengths = [len(pos_x), len(pos_y), len(pos_yaw), len(vel_vx), len(vel_vy), len(vel_wz)]
            if len(set(lengths)) > 1:
                raise ValueError(f"参考轨迹数据长度不一致: {lengths}")
            
            # 组合为N×6数组：[x, y, yaw, vx, vy, wz]
            ref_pos = np.column_stack([pos_x, pos_y, pos_yaw])
            ref_x = np.column_stack([vel_vx, vel_vy, vel_wz])
            _controller_state['ref_traj'] = np.hstack([ref_pos, ref_x])
            
            # 确保ref_traj是2维数组
            if _controller_state['ref_traj'].ndim == 1:
                if len(_controller_state['ref_traj']) % 6 == 0:
                    _controller_state['ref_traj'] = _controller_state['ref_traj'].reshape(-1, 6)
                else:
                    raise ValueError(f"参考轨迹数据长度不能被6整除: {len(_controller_state['ref_traj'])}")
            
            # 确保ref_traj的列数为6
            if _controller_state['ref_traj'].shape[1] != 6:
                raise ValueError(f"参考轨迹列数不正确，期望6列，实际{_controller_state['ref_traj'].shape[1]}列")
            
            print(f"[Python S-Function] 成功加载新格式参考轨迹，数据点数: {len(_controller_state['ref_traj'])}, 形状: {_controller_state['ref_traj'].shape}")
            
        except Exception as e:
            print(f"[Python S-Function] 警告：解析ref_trajectory结构体时出错: {e}")
            print(f"[Python S-Function] 尝试使用备用方法...")
            import traceback
            traceback.print_exc()
            raise
    
    # 兼容旧格式：直接包含position和velocity字段（N×3数组）
    elif 'position' in ref_data:
        print(f"[Python S-Function] 检测到Trucksim格式参考轨迹")
        ref_pos = ref_data['position']
        ref_x = ref_data['velocity']
        
        # 确保是numpy数组
        ref_pos = np.array(ref_pos)
        ref_x = np.array(ref_x)
        
        # 如果是1维数组，尝试重塑
        if ref_pos.ndim == 1:
            if len(ref_pos) % 3 == 0:
                ref_pos = ref_pos.reshape(-1, 3)
            else:
                raise ValueError(f"position数据长度不能被3整除: {len(ref_pos)}")
        if ref_x.ndim == 1:
            if len(ref_x) % 3 == 0:
                ref_x = ref_x.reshape(-1, 3)
            else:
                raise ValueError(f"velocity数据长度不能被3整除: {len(ref_x)}")
        
        # 检查维度匹配
        if ref_pos.shape[0] != ref_x.shape[0]:
            raise ValueError(f"position和velocity数据行数不匹配: {ref_pos.shape[0]} vs {ref_x.shape[0]}")
        if ref_pos.shape[1] != 3 or ref_x.shape[1] != 3:
            raise ValueError(f"position或velocity列数不正确: {ref_pos.shape[1]} vs {ref_x.shape[1]}")
        
        _controller_state['ref_traj'] = np.hstack([ref_pos, ref_x])
        
        # 确保ref_traj是2维数组
        if _controller_state['ref_traj'].ndim == 1:
            if len(_controller_state['ref_traj']) % 6 == 0:
                _controller_state['ref_traj'] = _controller_state['ref_traj'].reshape(-1, 6)
            else:
                raise ValueError(f"参考轨迹数据长度不能被6整除: {len(_controller_state['ref_traj'])}")
        
        print(f"[Python S-Function] 成功加载Trucksim格式参考轨迹，数据点数: {len(_controller_state['ref_traj'])}, 形状: {_controller_state['ref_traj'].shape}")
    
    # 兼容Carsim格式
    elif 'Pos' in ref_data:
        print(f"[Python S-Function] 检测到Carsim格式参考轨迹")
        ref_pos = ref_data['Pos']
        ref_x = ref_data['X']
        
        # 确保是numpy数组
        ref_pos = np.array(ref_pos)
        ref_x = np.array(ref_x)
        
        # 如果是1维数组，尝试重塑
        if ref_pos.ndim == 1:
            if len(ref_pos) % 3 == 0:
                ref_pos = ref_pos.reshape(-1, 3)
            else:
                raise ValueError(f"Pos数据长度不能被3整除: {len(ref_pos)}")
        if ref_x.ndim == 1:
            if len(ref_x) % 3 == 0:
                ref_x = ref_x.reshape(-1, 3)
            else:
                raise ValueError(f"X数据长度不能被3整除: {len(ref_x)}")
        
        # 检查维度匹配
        if ref_pos.shape[0] != ref_x.shape[0]:
            raise ValueError(f"Pos和X数据行数不匹配: {ref_pos.shape[0]} vs {ref_x.shape[0]}")
        if ref_pos.shape[1] != 3 or ref_x.shape[1] != 3:
            raise ValueError(f"Pos或X列数不正确: {ref_pos.shape[1]} vs {ref_x.shape[1]}")
        
        _controller_state['ref_traj'] = np.hstack([ref_pos, ref_x])
        
        # 确保ref_traj是2维数组
        if _controller_state['ref_traj'].ndim == 1:
            if len(_controller_state['ref_traj']) % 6 == 0:
                _controller_state['ref_traj'] = _controller_state['ref_traj'].reshape(-1, 6)
            else:
                raise ValueError(f"参考轨迹数据长度不能被6整除: {len(_controller_state['ref_traj'])}")
        
        print(f"[Python S-Function] 成功加载Carsim格式参考轨迹，数据点数: {len(_controller_state['ref_traj'])}, 形状: {_controller_state['ref_traj'].shape}")
    
    else:
        available_keys = [k for k in ref_data.keys() if not k.startswith('__')]
        raise KeyError(
            f"参考轨迹数据格式不支持。\n"
            f"期望格式之一：\n"
            f"  1. ref_trajectory结构体（包含position和velocity子结构体）\n"
            f"  2. 'position'和'velocity'字段（N×3数组）\n"
            f"  3. 'Pos'和'X'字段（N×3数组）\n"
            f"可用字段: {available_keys}"
        )
    
    # 设置参数（确保是整数，因为用作数组索引）
    _controller_state['Np'] = int(Np)
    _controller_state['Nc'] = int(Nc)
    _controller_state['sample_interval'] = int(sample_interval)  # 1 与 Koopman 步长对齐
    _controller_state['decimation'] = int(decimation)
    
    # 重置所有状态变量（重要：确保每次初始化都从干净状态开始）
    # 注意：如果之前运行过仿真，Python模块的全局状态会被保留
    # 因此必须在初始化时重置所有状态，否则index会从上次的值继续累加
    old_index = _controller_state['index']  # 记录旧值用于调试
    # 初始归一化控制：0.5 对应零转矩/零转角
    _controller_state['u_prev'] = 0.5 * np.ones(12)
    _controller_state['nearest_idx'] = 1
    _controller_state['start_idx'] = 1
    _controller_state['index'] = 0
    _controller_state['mpc_step'] = 0
    _controller_state['bad_count'] = 0
    _controller_state['last_state'] = None
    _controller_state['u_init_prev'] = None

    # 初始化零阶保持输出
    try:
        _controller_state['last_control_output'] = convert_to_control_output(
            _controller_state['u_prev']
        ).reshape(12, 1)
    except Exception:
        _controller_state['last_control_output'] = np.zeros(12).reshape(12, 1)

    if old_index > 0:
        print(f"[Python S-Function] 警告：检测到之前的状态 (index={old_index})，已重置为0")

    ref_traj_len = len(_controller_state['ref_traj'])
    trajectory_end_threshold = int(Np * sample_interval + 200)
    recommended_steps = max(0, ref_traj_len - trajectory_end_threshold)
    recommended_sim_time = recommended_steps * 0.01  # 10ms 步长对齐

    print(f"[Python S-Function] 初始化完成 (Koopman-MPC V2)")
    print(f"  参考轨迹长度: {ref_traj_len}, Np={Np}, sample_interval={sample_interval}")
    print(f"  控制抽取倍率: decimation={_controller_state['decimation']} (每{_controller_state['decimation']}次调用求解一次MPC)")
    print(f"  推荐仿真时间: {recommended_sim_time:.1f}秒")
    
    return True


def compute_control(state_input):
    """
    计算MPC控制输出（主要接口函数）
    
    Args:
        state_input: 车辆状态 [X(m), Y(m), Yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]
                     Trucksim输出已经是国际标准单位，无需转换
    
    Returns:
        control_output: 控制信号 [12维]
                        [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
                         steer_LR(deg), steer_RR(deg),
                         torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
                         torque_LR(N·m), torque_RR(N·m)]
    """
    global _controller_state
    
    if _controller_state['encoder'] is None:
        print("[Python S-Function] Warning: Controller not initialized, returning zero control")
        return np.zeros(12).reshape(12, 1)

    try:
        import torch
        encoder = _controller_state['encoder']
        mpc = _controller_state['mpc']
        ref_traj = _controller_state['ref_traj']
        device = _controller_state['device']

        if isinstance(state_input, np.ndarray):
            state_arr = state_input
        else:
            state_arr = np.array(state_input, dtype=float)

        x_cur = np.array([
            float(state_arr[0]), float(state_arr[1]), float(state_arr[2]),
            float(state_arr[3]), float(state_arr[4]), float(state_arr[5])
        ])

        _controller_state['index'] += 1
        call_idx = int(_controller_state['index'])
        decimation = int(_controller_state.get('decimation', 10))
        if decimation <= 0:
            decimation = 10

        do_solve = ((call_idx - 1) % decimation == 0)
        if not do_solve:
            return _controller_state.get('last_control_output', np.zeros(12).reshape(12, 1))

        _controller_state['mpc_step'] = int(_controller_state.get('mpc_step', 0)) + 1
        mpc_step = int(_controller_state['mpc_step'])

        if mpc_step <= 10 or mpc_step % 100 == 0:
            print(f"\n[MPC输入] Step {mpc_step}:")
            print(f"  车辆状态: X={x_cur[0]:.4f}m, Y={x_cur[1]:.4f}m, Yaw={x_cur[2]*180/np.pi:.4f}°")
            nearest_idx_dbg = int(_controller_state['nearest_idx'])
            print(f"  参考轨迹索引: nearest_idx={nearest_idx_dbg}, 轨迹长度={len(ref_traj)}")

        if mpc_step == 1:
            _controller_state['start_idx'], _ = find_nearest_point(ref_traj, x_cur, 1, len(ref_traj))
            _controller_state['start_idx'] = int(_controller_state['start_idx'])
            _controller_state['nearest_idx'] = int(_controller_state['start_idx'])
            _controller_state['bad_count'] = 0
            _controller_state['last_state'] = x_cur.copy()
            print(f"[Python S-Function] 初始化完成: start_idx={_controller_state['start_idx']}")

        if _controller_state['last_state'] is not None:
            if np.sum(np.abs(_controller_state['last_state'] - x_cur)) < 0.001:
                _controller_state['bad_count'] += 1
            else:
                _controller_state['bad_count'] = 0
        _controller_state['last_state'] = x_cur.copy()

        current_idx = int(_controller_state['nearest_idx'])
        tmp_nearest_idx, _ = find_nearest_point(ref_traj, x_cur, current_idx, 500)
        new_idx = max(current_idx, int(tmp_nearest_idx))
        if _controller_state['bad_count'] > 30 and new_idx == current_idx:
            new_idx = min(current_idx + 1, len(ref_traj) - 1)
        _controller_state['nearest_idx'] = new_idx

        nearest_idx = int(_controller_state['nearest_idx'])
        sample_interval = int(_controller_state['sample_interval'])
        Np = int(_controller_state['Np'])

        if ref_traj is None:
            return np.zeros(12).reshape(12, 1)
        ref_traj = np.array(ref_traj)
        if ref_traj.ndim == 1:
            if len(ref_traj) % 6 == 0:
                ref_traj = ref_traj.reshape(-1, 6)
            else:
                return np.zeros(12).reshape(12, 1)
        if ref_traj.shape[1] != 6:
            return np.zeros(12).reshape(12, 1)

        trajectory_end_threshold = int(Np * sample_interval + 200)
        if nearest_idx >= len(ref_traj) - trajectory_end_threshold:
            print(f"[Python S-Function] 到达轨迹终点，停止跟踪")
            return np.zeros(12).reshape(12, 1)
        if _controller_state['bad_count'] > 50:
            print(f"[Python S-Function] 车辆卡住，停止跟踪")
            return np.zeros(12).reshape(12, 1)

        nearest_idx = max(0, min(nearest_idx, len(ref_traj) - 1))
        end_idx = int(min(nearest_idx + Np * sample_interval, len(ref_traj)))
        temp_refr = ref_traj[nearest_idx:end_idx:sample_interval, :]

        if len(temp_refr) < Np:
            last_point = temp_refr[-1, :] if len(temp_refr) > 0 else ref_traj[-1, :]
            padding = np.tile(last_point, (int(Np - len(temp_refr)), 1))
            temp_refr = np.vstack([temp_refr, padding])
        elif len(temp_refr) > Np:
            temp_refr = temp_refr[:Np, :]

        if temp_refr.ndim == 1:
            temp_refr = temp_refr.reshape(1, -1)
        if temp_refr.shape[0] == 0 or temp_refr.shape[1] < 6:
            return np.zeros(12).reshape(12, 1)

        ref_r_6 = temp_refr.astype(np.float32)

        # -------- 实际位置与参考位置差异（便于调试，前 10 步 + 每 100 步输出）--------
        if mpc_step <= 10 or mpc_step % 100 == 0:
            ref_first = ref_r_6[0]
            dx = float(x_cur[0] - ref_first[0])
            dy = float(x_cur[1] - ref_first[1])
            dyaw_rad = float(x_cur[2] - ref_first[2])
            dyaw_deg = dyaw_rad * 180.0 / np.pi
            pos_err_norm = np.sqrt(dx * dx + dy * dy)
            dvx = float(x_cur[3] - ref_first[3])
            dvy = float(x_cur[4] - ref_first[4])
            dwz = float(x_cur[5] - ref_first[5])
            print(f"[位置/参考] Step {mpc_step}: 实际(x,y,yaw)=({x_cur[0]:.4f}, {x_cur[1]:.4f}, {x_cur[2]*180/np.pi:.4f}°)  "
                  f"参考首点(x,y,yaw)=({ref_first[0]:.4f}, {ref_first[1]:.4f}, {ref_first[2]*180/np.pi:.4f}°)")
            print(f"  差异: Δx={dx:.4f}m, Δy={dy:.4f}m, Δyaw={dyaw_deg:.4f}°, 位置误差(范数)={pos_err_norm:.4f}m  "
                  f"Δvx={dvx:.4f}, Δvy={dvy:.4f}, Δwz={dwz:.6f}")

        # 与 mpc_dk 一致：输入给网络的 ref 的 x、y 为相对当前车坐标 (ref_x - vehicle_x, ref_y - vehicle_y)
        # 当前状态编码用 (0, 0, yaw, vx, vy, wz)；ref 仅 x/y 相对当前，yaw/vx/vy/wz 保持原样
        x_cur_for_enc = np.array(x_cur, dtype=np.float32)
        x_cur_for_enc[0] = 0.0
        x_cur_for_enc[1] = 0.0
        ref_r_6_rel = ref_r_6.copy()
        ref_r_6_rel[:, 0] -= float(x_cur[0])  # ref x 相对当前车
        ref_r_6_rel[:, 1] -= float(x_cur[1])  # ref y 相对当前车

        z0 = encode_state(encoder, x_cur_for_enc, device)
        z_ref = encode_ref_trajectory(encoder, ref_r_6_rel, device)

        # -------- DEBUG: 编码器前的状态与编码后（前若干步）--------
        if mpc_step <= 5:
            print(f"[MPC DEBUG] Step {mpc_step}: x_cur(原始)={x_cur.round(6).tolist()}")
            print(f"  编码器前 当前状态 x_cur_for_enc={x_cur_for_enc.round(6).tolist()}")
            print(f"  编码器前 参考首点 ref_r_6_rel[0]={ref_r_6_rel[0].round(6).tolist()}")
            print(f"  编码器前 参考末点 ref_r_6_rel[-1]={ref_r_6_rel[-1].round(6).tolist()}")
            print(f"  编码后 z0[:6]={np.array(z0[:6]).round(6).tolist()}, z_ref[0][:6]={z_ref[0][:6].round(6).tolist()}")

        z0_t = torch.from_numpy(z0).float().unsqueeze(0).to(device)
        z_ref_t = torch.from_numpy(z_ref).float().unsqueeze(0).to(device)

        u_prev = _controller_state['u_prev']
        u_init = _controller_state.get('u_init_prev')
        if u_init is not None and u_init.shape[1] == z_ref.shape[0]:
            u_init_t = torch.from_numpy(u_init).float().unsqueeze(0).to(device)
        else:
            u_init_t = None

        U_opt = mpc.optimize(z0_t, z_ref_t, u_init=u_init_t)
        u_new = U_opt[0, 0, :].detach().cpu().numpy()
        _controller_state['u_prev'] = np.clip(u_new, 0.0, 1.0)
        _controller_state['u_init_prev'] = U_opt[0].detach().cpu().numpy()

        control_output = convert_to_control_output(_controller_state['u_prev'])
        _controller_state['last_control_output'] = control_output.reshape(12, 1)

        if mpc_step <= 10 or mpc_step % 100 == 0:
            print(f"[MPC控制] Step {mpc_step}: 实际控制 steer(deg)={control_output[:6].round(3).tolist()}, torque(N·m)={control_output[6:].round(2).tolist()}")

        return _controller_state['last_control_output']

    except Exception as e:
        print(f"[Python S-Function] Error in compute_control: {e}, returning zero control")
        return np.zeros(12).reshape(12, 1)


def reset_controller():
    """重置控制器状态"""
    global _controller_state
    _controller_state['u_prev'] = 0.5 * np.ones(12)
    _controller_state['nearest_idx'] = 1
    _controller_state['start_idx'] = 1
    _controller_state['index'] = 0
    _controller_state['mpc_step'] = 0
    _controller_state['bad_count'] = 0
    _controller_state['last_state'] = None
    _controller_state['u_init_prev'] = None
    try:
        _controller_state['last_control_output'] = convert_to_control_output(
            _controller_state['u_prev']
        ).reshape(12, 1)
    except Exception:
        _controller_state['last_control_output'] = np.zeros(12).reshape(12, 1)
    print("[Python S-Function] 控制器状态已重置 (Koopman-MPC V2)")


# 如果作为模块导入，提供标准接口
if __name__ != '__main__':
    # 这些函数可以被MATLAB直接调用
    pass

