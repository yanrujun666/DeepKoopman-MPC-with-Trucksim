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

# 导入MPC控制器
try:
    from ddk_controller import DeepEDMD, MPCController
except ImportError:
    # 如果导入失败，再次尝试添加路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from ddk_controller import DeepEDMD, MPCController


# 全局状态（在S-Function调用之间保持）
_controller_state = {
    'deepedmd': None,
    'mpc': None,
    'ref_traj': None,
    # 注意：控制量采用CONTROL_MIN/MAX做min-max归一化
    # 因此 u=0.5 对应“零转矩/零转角”（区间中心），u=0 对应最小转矩/最小转角
    'u_prev': 0.5 * np.ones(12),  # 12维控制：前6转矩 + 后6转向角（归一化后）
    'nearest_idx': 1,
    'start_idx': 1,
    # index: Python侧调用计数（当前Simulink/Trucksim为1ms步长，会非常快）
    'index': 0,
    # mpc_step: 实际MPC更新计数（按目标采样周期，例如50ms）
    'mpc_step': 0,
    # 控制抽取倍率：每 decimation 次调用才真正执行一次MPC求解
    # 例如：仿真步长1ms、MPC采样50ms，则 decimation=50
    'decimation': 50,
    # 上一次输出（用于零阶保持），格式为(12,1)
    'last_control_output': np.zeros(12).reshape(12, 1),
    'bad_count': 0,
    'last_state': None,
    'sample_interval': 5,
    'Np': 30,
    'Nc': 30
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


def initialize_controller(param_path, data_path, Np=30, Nc=30, sample_interval=5, decimation=50):
    """
    初始化MPC控制器
    
    Args:
        param_path: DeepEDMD模型参数文件路径（必须是.pth格式）
        data_path: 参考数据文件路径
        Np: 预测时域
        Nc: 控制时域
        sample_interval: 采样间隔
    """
    global _controller_state
    
    print(f"[Python S-Function] 初始化控制器...")
    print(f"  参数文件: {param_path}")
    print(f"  数据文件: {data_path}")
    
    # 加载DeepEDMD模型
    _controller_state['deepedmd'] = DeepEDMD(param_path)
    
    # 创建MPC控制器
    # Trucksim控制维度为12维（6个转矩+6个转向角）
    # ========= 调参（更平顺：速度优先 + 转矩更平滑）=========
    # Q：状态跟踪权重（6维：[X, Y, Yaw, vx, vy, yaw_rate]）
    # - 速度(vx)优先，但避免极端大权重导致饱和
    Q = np.diag([20, 80, 120, 300, 40, 40])
    # ========= 将控制增量降维到2维： [delta_torque_scalar, delta_front_steer_scalar] =========
    # 说明：
    # - MPC内部求解的变量维度为2（扭矩标量 + 前轮转角标量）
    # - 通过 B_ 将2维增量映射回12维增量（与训练数据集执行器结构一致）：
    #   torques(6)=delta_torque
    #   steer: LF/RF=+delta_steer, LM/RM=0, LR/RR=-delta_steer
    B_delta = np.array(
        [[1, 0],
         [1, 0],
         [1, 0],
         [1, 0],
         [1, 0],
         [1, 0],
         [0, 1],
         [0, 1],
         [0, 0],
         [0, 0],
         [0, -1],
         [0, -1]], dtype=float
    )

    # R：控制增量权重（2维：[torque_scalar, steer_scalar]）
    R = np.diag([800, 12000])
    # delta_umax：2维控制单步增量上限（归一化）
    delta_umax = np.array([0.05, 0.02])
    
    # 时间步长配置
    # DeepEDMD模型的时间步长：0.01s（10ms）
    # MPC采样时间：sample_interval * 0.01s
    model_dt = 0.01  # 模型时间步长（秒）
    mpc_dt = float(sample_interval) * model_dt  # MPC采样时间（秒）
    
    _controller_state['mpc'] = MPCController(
        _controller_state['deepedmd'],
        Np=Np,
        Nc=Nc,
        Q=Q,
        R=R,
        delta_umax=delta_umax,
        model_dt=model_dt,
        mpc_dt=mpc_dt,
        control_delta_mapping=B_delta
    )
    
    # 启用跟踪误差打印（在需要时动态控制）
    
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
    _controller_state['sample_interval'] = int(sample_interval)
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

    # 初始化零阶保持输出（避免非更新步返回空值）
    try:
        _controller_state['last_control_output'] = _controller_state['mpc'].convert_to_control_output(
            _controller_state['u_prev']
        ).reshape(12, 1)
    except Exception:
        _controller_state['last_control_output'] = np.zeros(12).reshape(12, 1)
    
    if old_index > 0:
        print(f"[Python S-Function] 警告：检测到之前的状态 (index={old_index})，已重置为0")
    
    # 计算推荐仿真时间
    ref_traj_len = len(_controller_state['ref_traj'])
    trajectory_end_threshold = int(Np * sample_interval + 200)
    recommended_steps = max(0, ref_traj_len - trajectory_end_threshold)
    recommended_sim_time = recommended_steps / 20.0  # 基于50ms采样间隔
    
    print(f"[Python S-Function] 初始化完成")
    print(f"  参考轨迹长度: {ref_traj_len}, Np={Np}, Nc={Nc}, sample_interval={sample_interval}")
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
    
    # 如果控制器未初始化，返回零控制（避免抛出异常，因为MATLAB代码生成不支持try-catch）
    if _controller_state['deepedmd'] is None:
        print("[Python S-Function] Warning: Controller not initialized, returning zero control")
        # 返回列向量（12x1），确保MATLAB能正确接收
        return np.zeros(12).reshape(12, 1)
    
    try:
        deepedmd = _controller_state['deepedmd']
        mpc = _controller_state['mpc']
        ref_traj = _controller_state['ref_traj']
        
        # Trucksim输出已经是国际标准单位，不需要单位转换
        # 状态格式: [X(m), Y(m), Yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]
        # 如果已经是numpy数组，直接使用；否则转换为数组
        if isinstance(state_input, np.ndarray):
            state_arr = state_input
        else:
            # 处理MATLAB数组或Python列表
            state_arr = np.array(state_input, dtype=float)
        
        x_cur = np.array([
            float(state_arr[0]),                    # X (m) - 已经是国际标准单位
            float(state_arr[1]),                    # Y (m) - 已经是国际标准单位
            float(state_arr[2]),                    # Yaw (rad) - 已经是国际标准单位
            float(state_arr[3]),                    # vx (m/s) - 已经是国际标准单位
            float(state_arr[4]),                    # vy (m/s) - 已经是国际标准单位
            float(state_arr[5])                     # yaw_rate (rad/s) - 已经是国际标准单位
        ])
        
        # 1ms调用计数
        _controller_state['index'] += 1
        call_idx = int(_controller_state['index'])

        # ====== 50ms门控：仅每 decimation 次调用求解一次MPC，其余时间零阶保持 ======
        decimation = int(_controller_state.get('decimation', 50))
        if decimation <= 0:
            decimation = 50

        do_solve = ((call_idx - 1) % decimation == 0)
        if not do_solve:
            # 非MPC更新步：直接返回上一次控制输出（零阶保持）
            return _controller_state.get('last_control_output', np.zeros(12).reshape(12, 1))

        # MPC更新计数（用于打印“Step”）
        _controller_state['mpc_step'] = int(_controller_state.get('mpc_step', 0)) + 1
        mpc_step = int(_controller_state['mpc_step'])

        # 打印输入信息（前10个MPC步或每100个MPC步）
        if mpc_step <= 10 or mpc_step % 100 == 0:
            print(f"\n[MPC输入] Step {mpc_step}:")
            print(f"  车辆状态: X={x_cur[0]:.4f}m, Y={x_cur[1]:.4f}m, Yaw={x_cur[2]*180/np.pi:.4f}°")
            print(f"  速度: vx={x_cur[3]:.4f}m/s, vy={x_cur[4]:.4f}m/s, yaw_rate={x_cur[5]*180/np.pi:.4f}°/s")
            nearest_idx_dbg = int(_controller_state['nearest_idx'])
            print(f"  参考轨迹索引: nearest_idx={nearest_idx_dbg}, 轨迹长度={len(ref_traj)}")
            if 0 <= nearest_idx_dbg < len(ref_traj):
                ref_state = ref_traj[nearest_idx_dbg, :]
                print(f"  参考状态: X={ref_state[0]:.4f}m, Y={ref_state[1]:.4f}m, Yaw={ref_state[2]*180/np.pi:.4f}°")
                print(f"  参考速度: vx={ref_state[3]:.4f}m/s, vy={ref_state[4]:.4f}m/s, yaw_rate={ref_state[5]*180/np.pi:.4f}°/s")
        
        # 初始化最近点索引（在第1个MPC步做一次全局搜索）
        if mpc_step == 1:
            _controller_state['start_idx'], _ = find_nearest_point(
                ref_traj, x_cur, 1, len(ref_traj)
            )
            _controller_state['start_idx'] = int(_controller_state['start_idx'])
            _controller_state['nearest_idx'] = int(_controller_state['start_idx'])
            _controller_state['bad_count'] = 0
            _controller_state['last_state'] = x_cur.copy()
            print(f"[Python S-Function] 初始化完成: start_idx={_controller_state['start_idx']}")
        
        # 如果还没有初始化（idx == 1），使用默认值，但此时nearest_idx应该是1
        # 在MATLAB代码中，index=1时nearest_idx=1（默认值），所以可以正常工作
        
        # 检测状态是否卡住
        if _controller_state['last_state'] is not None:
            if np.sum(np.abs(_controller_state['last_state'] - x_cur)) < 0.001:
                _controller_state['bad_count'] += 1
            else:
                _controller_state['bad_count'] = 0
        _controller_state['last_state'] = x_cur.copy()
        
        # 更新最近点索引（对应MATLAB代码：nearest_idx = max(nearest_idx, tmp_nearest_idx)）
        # 重要：MATLAB代码中索引只能向前或保持不变，不能后退
        # 这确保了参考轨迹始终向前推进，避免索引卡住
        current_idx = int(_controller_state['nearest_idx'])
        tmp_nearest_idx, min_dis = find_nearest_point(
            ref_traj, x_cur, current_idx, 500  # 从当前索引开始向前搜索
        )
        new_idx = max(current_idx, int(tmp_nearest_idx))
        
        # 如果车辆卡住且最近点索引长时间不变，强制推进索引
        if _controller_state['bad_count'] > 30 and new_idx == current_idx:
            new_idx = min(current_idx + 1, len(ref_traj) - 1)
            if _controller_state['bad_count'] % 50 == 0:
                print(f"[DEBUG] 强制推进索引: {current_idx} -> {new_idx}")
        
        _controller_state['nearest_idx'] = new_idx
        
        # 提取参考轨迹（确保所有索引都是整数）
        nearest_idx = int(_controller_state['nearest_idx'])
        sample_interval = int(_controller_state['sample_interval'])
        Np = int(_controller_state['Np'])
        
        # 检查ref_traj是否有效
        if ref_traj is None:
            print(f"[Python S-Function] 错误：参考轨迹未加载")
            return np.zeros(12).reshape(12, 1)

        # 确保ref_traj是numpy数组
        ref_traj = np.array(ref_traj)

        # 确保ref_traj是2维数组
        if ref_traj.ndim == 1:
            print(f"[Python S-Function] 警告：参考轨迹是1维数组，尝试重塑为2维")
            if len(ref_traj) % 6 == 0:
                ref_traj = ref_traj.reshape(-1, 6)
            else:
                print(f"[Python S-Function] 错误：参考轨迹长度不能被6整除")
                return np.zeros(12).reshape(12, 1)
        
        # 检查ref_traj的列数
        if ref_traj.shape[1] != 6:
            print(f"[Python S-Function] 错误：参考轨迹列数不正确，期望6列，实际{ref_traj.shape[1]}列")
            return np.zeros(12).reshape(12, 1)
        
        # 检查是否接近轨迹终点或车辆卡住
        trajectory_end_threshold = int(Np * sample_interval + 200)
        if nearest_idx >= len(ref_traj) - trajectory_end_threshold:
            print(f"[Python S-Function] 到达轨迹终点，停止跟踪")
            return np.zeros(12).reshape(12, 1)
        
        if _controller_state['bad_count'] > 50:
            print(f"[Python S-Function] 车辆卡住，停止跟踪")
            return np.zeros(12).reshape(12, 1)
        
        # 确保nearest_idx在有效范围内
        if nearest_idx < 0:
            nearest_idx = 0
        if nearest_idx >= len(ref_traj):
            nearest_idx = len(ref_traj) - 1
        
        # 参考轨迹提取：对应MATLAB代码 r(nearest_idx:sample_interval:Np*sample_interval+nearest_idx-1, :)
        # MATLAB切片：从nearest_idx开始，每隔sample_interval取一个点，到nearest_idx+Np*sample_interval-1结束（包含）
        # 这意味着要提取Np个点：nearest_idx, nearest_idx+sample_interval, ..., nearest_idx+(Np-1)*sample_interval
        # Python切片：nearest_idx:end_idx:sample_interval，其中end_idx不包含
        # 所以end_idx应该是 nearest_idx + Np*sample_interval（不包含，所以最后一个索引是nearest_idx+(Np-1)*sample_interval）
        end_idx = int(min(nearest_idx + Np * sample_interval, len(ref_traj)))
        temp_refr = ref_traj[nearest_idx:end_idx:sample_interval, :]
        
        # 确保提取的点数严格等于Np
        if len(temp_refr) < Np:
            last_point = temp_refr[-1, :] if len(temp_refr) > 0 else ref_traj[-1, :]
            padding = np.tile(last_point, (int(Np - len(temp_refr)), 1))
            temp_refr = np.vstack([temp_refr, padding])
        elif len(temp_refr) > Np:
            temp_refr = temp_refr[:Np, :]
        
        # 检查temp_refr是否有效
        if temp_refr is None or len(temp_refr) == 0:
            print(f"[Python S-Function] 错误：提取的参考轨迹为空")
            return np.zeros(12).reshape(12, 1)
        
        # 确保temp_refr是2维数组且至少有1行
        if temp_refr.ndim == 1:
            temp_refr = temp_refr.reshape(1, -1)
        
        # 检查temp_refr的第一行是否有效
        if temp_refr.shape[0] == 0 or temp_refr.shape[1] < 3:
            print(f"[Python S-Function] 错误：参考轨迹格式不正确，形状={temp_refr.shape}")
            return np.zeros(12).reshape(12, 1)
        
        # 使用全局坐标（模型在全局坐标系中训练，不需要坐标变换）
        # 直接使用全局参考轨迹，不进行坐标变换
        ref_r = temp_refr.copy()
        
        # 直接使用全局状态进行编码（不进行坐标变换）
        # 模型在全局坐标系中训练，编码器内部有BatchNorm
        x_lift = deepedmd.encoder(x_cur)
        
        # 求解MPC（仅在需要时打印跟踪误差）
        print_tracking_error = (mpc_step <= 10 or mpc_step % 100 == 0)
        mpc._print_tracking_error = print_tracking_error
        # MPC内部已将控制增量降维到2维（通过control_delta_mapping映射回12维）
        # 这里直接传入全12维u_prev（归一化），solve返回2维delta_u
        delta_u_2, success = mpc.solve(x_lift, _controller_state['u_prev'], ref_r)
        
        if success:
            # 更新全12维控制：u_new_full = u_prev_full + B_u_map * delta_u_2
            delta_u_full = mpc.B_u_map @ delta_u_2
            u_new = _controller_state['u_prev'] + delta_u_full
            _controller_state['u_prev'] = np.clip(u_new, 0.0, 1.0)
            
        else:
            print(f"[Python S-Function] Warning: MPC求解失败，保持上一时刻控制")
        
        # 转换为实际控制指令（使用统一方法）
        control_output = mpc.convert_to_control_output(_controller_state['u_prev'])
        # 缓存控制输出（用于后续49个非更新步的零阶保持）
        _controller_state['last_control_output'] = control_output.reshape(12, 1)
        
        # 打印控制信息（前10步或每100步）
        if mpc_step <= 10 or mpc_step % 100 == 0:
            print(f"[MPC控制] Step {mpc_step}:")
            u_prev_str = ', '.join([f"{_controller_state['u_prev'][i]:.4f}" for i in range(12)])
            delta_u_str = ', '.join([f"{delta_u_2[i]:.4f}" for i in range(len(delta_u_2))])
            print(f"  归一化控制: u_prev=[{u_prev_str}]")
            print(f"  控制增量(delta_u_2): [{delta_u_str}]")
            print(f"  实际控制: steer_LF={control_output[0]:.3f}°,steer_RF={control_output[1]:.3f}°,steer_LM={control_output[2]:.3f}°,steer_RM={control_output[3]:.3f}°,steer_LR={control_output[4]:.3f}°,steer_RR={control_output[5]:.3f}°, \
                torque_LF={control_output[6]:.2f}N·m, torque_RF={control_output[7]:.2f}N·m, torque_LM={control_output[8]:.2f}N·m, torque_RM={control_output[9]:.2f}N·m, torque_LR={control_output[10]:.2f}N·m, torque_RR={control_output[11]:.2f}N·m")
            print(f"  求解状态: {'成功' if success else '失败'}")
        
        # 返回列向量（12x1），确保MATLAB能正确接收
        return _controller_state['last_control_output']
    
    except Exception as e:
        # 如果发生任何错误，返回零控制（避免MATLAB代码生成问题）
        print(f"[Python S-Function] Error in compute_control: {e}, returning zero control")
        # 返回列向量（12x1），确保MATLAB能正确接收
        return np.zeros(12).reshape(12, 1)


def reset_controller():
    """重置控制器状态"""
    global _controller_state
    # 初始归一化控制：0.5 对应零转矩/零转角
    _controller_state['u_prev'] = 0.5 * np.ones(12)
    _controller_state['nearest_idx'] = 1
    _controller_state['start_idx'] = 1
    _controller_state['index'] = 0
    _controller_state['mpc_step'] = 0
    _controller_state['bad_count'] = 0
    _controller_state['last_state'] = None
    try:
        if _controller_state.get('mpc') is not None:
            _controller_state['last_control_output'] = _controller_state['mpc'].convert_to_control_output(
                _controller_state['u_prev']
            ).reshape(12, 1)
        else:
            _controller_state['last_control_output'] = np.zeros(12).reshape(12, 1)
    except Exception:
        _controller_state['last_control_output'] = np.zeros(12).reshape(12, 1)
    print("[Python S-Function] 控制器状态已重置")


# 如果作为模块导入，提供标准接口
if __name__ != '__main__':
    # 这些函数可以被MATLAB直接调用
    pass

