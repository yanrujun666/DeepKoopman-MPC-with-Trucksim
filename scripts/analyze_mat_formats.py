"""
分析参考轨迹 .mat 与训练集 .mat 的格式差异，为「训练集 -> 参考轨迹」转换做准备。

参考轨迹来源：
  - 生成脚本: MPC/ref_trajectory/generate_straight_acceleration_trajectory.m
  - 示例文件: MPC/ref_trajectory/straight_acceleration_trajectory_ref.mat

训练集示例：
  - data/all/all_wheel_steer_Scenario_snake_acc_5m_s.mat
"""

import os
import sys
import numpy as np
import scipy.io as scio

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 参考轨迹 .mat（MPC 使用的格式）
REF_MAT = os.path.join(ROOT, "MPC", "ref_trajectory", "straight_acceleration_trajectory_ref.mat")
# 训练集 .mat
DATA_MAT = os.path.join(ROOT, "data", "all", "all_wheel_steer_Scenario_snake_acc_5m_s.mat")


def describe_value(val, indent=0, prefix=""):
    """递归描述变量：类型、shape、dtype、若有字段则列出字段。"""
    pad = "  " * indent
    lines = []

    if isinstance(val, np.ndarray):
        if val.dtype.names:
            # 结构化数组（MATLAB struct）
            lines.append(f"{pad}{prefix}ndarray (structured), shape={val.shape}, dtype.names={val.dtype.names}")
            if val.size == 1 and val.ndim == 0:
                for name in val.dtype.names:
                    sub = val[name]
                    lines.extend(describe_value(sub, indent + 1, f"{name}: ").split("\n"))
            elif val.size > 0:
                # 取第一个元素看结构
                first = np.asarray(val).flat[0]
                lines.append(f"{pad}  (element 0 structure)")
                for name in val.dtype.names:
                    sub = first[name]
                    lines.extend(describe_value(sub, indent + 2, f"{name}: ").split("\n"))
        else:
            lines.append(f"{pad}{prefix}ndarray, shape={val.shape}, dtype={val.dtype}")
            if val.size <= 12 and val.ndim <= 1:
                lines.append(f"{pad}  sample: {val.tolist()}")
            elif val.ndim == 1 and len(val) > 0:
                lines.append(f"{pad}  sample (first 5): {val.ravel()[:5].tolist()}")
            elif val.ndim == 2 and val.size > 0:
                lines.append(f"{pad}  sample (first row): {val[0].tolist()}")
    elif isinstance(val, dict):
        lines.append(f"{pad}{prefix}dict, keys={list(val.keys())}")
        for k, v in val.items():
            if k.startswith("__"):
                continue
            lines.extend(describe_value(v, indent + 1, f"{k}: ").split("\n"))
    elif hasattr(val, "dtype") and hasattr(val, "shape"):
        lines.append(f"{pad}{prefix}array-like, shape={getattr(val, 'shape', '?')}, dtype={getattr(val, 'dtype', '?')}")
    else:
        lines.append(f"{pad}{prefix}{type(val).__name__} = {repr(val)[:80]}")
    return "\n".join(lines)


def extract_struct_field(struct_data, field_name):
    """从 MATLAB 结构体（numpy 结构化数组）中提取字段，与 ddk_mpc_sfunction 一致。"""
    if isinstance(struct_data, np.ndarray) and struct_data.dtype.names:
        if struct_data.shape == ():
            if field_name in struct_data.dtype.names:
                value = struct_data[field_name]
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    return value.item()
                return value
            raise KeyError(f"结构体中未找到字段 '{field_name}'，可用: {struct_data.dtype.names}")
        return np.array([struct_data[i][field_name] for i in range(len(struct_data))])
    if hasattr(struct_data, field_name):
        return getattr(struct_data, field_name)
    raise ValueError(f"无法提取 '{field_name}'，类型: {type(struct_data)}")


def analyze_ref_mat(path):
    """分析参考轨迹 .mat（与 MPC 加载方式一致：squeeze_me=True）。"""
    print("\n" + "=" * 60)
    print("1. 参考轨迹 .mat 格式（MPC 用）")
    print("=" * 60)
    if not os.path.isfile(path):
        print(f"文件不存在: {path}")
        return None
    data = scio.loadmat(path, squeeze_me=True)
    # 去掉 __header__ 等
    keys = [k for k in data.keys() if not k.startswith("__")]
    print(f"顶层变量: {keys}\n")
    for k in keys:
        print(describe_value(data[k], indent=0, prefix=f"{k}: "))
        print()
    # 若存在 ref_trajectory，解析出与 MPC 一致的 N×6 形式
    if "ref_trajectory" in data:
        try:
            ref = data["ref_trajectory"]
            pos_s = extract_struct_field(ref, "position")
            vel_s = extract_struct_field(ref, "velocity")
            time_arr = extract_struct_field(ref, "time")
            x = np.array(extract_struct_field(pos_s, "x")).flatten()
            y = np.array(extract_struct_field(pos_s, "y")).flatten()
            yaw = np.array(extract_struct_field(pos_s, "yaw")).flatten()
            vx = np.array(extract_struct_field(vel_s, "vx")).flatten()
            vy = np.array(extract_struct_field(vel_s, "vy")).flatten()
            wz = np.array(extract_struct_field(vel_s, "wz")).flatten()
            ref_6 = np.column_stack([x, y, yaw, vx, vy, wz])
            print("解析后的 ref_traj (N×6) [x, y, yaw, vx, vy, wz]:")
            print(f"  shape = {ref_6.shape}")
            print(f"  time length = {len(time_arr)}, dt sample = {np.diff(time_arr[:5])}")
            print(f"  前 3 行:\n{ref_6[:3]}")
        except Exception as e:
            print(f"解析 ref_trajectory 时出错: {e}")
    return data


def analyze_dataset_mat(path):
    """分析训练集 .mat（与 eval_sequence 一致：simplify_cells=True）。"""
    print("\n" + "=" * 60)
    print("2. 训练集 .mat 格式")
    print("=" * 60)
    if not os.path.isfile(path):
        print(f"文件不存在: {path}")
        return None
    data = scio.loadmat(path, simplify_cells=True)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print(f"顶层变量: {keys}\n")
    for k in keys:
        v = data[k]
        print(describe_value(v, indent=0, prefix=f"{k}: "))
        print()
    # 若存在 Vehicle_state_trucksim_39d，说明轨迹对应前 6 列
    if "Vehicle_state_trucksim_39d" in data:
        V = np.asarray(data["Vehicle_state_trucksim_39d"])
        print("轨迹相关列 (Vehicle_state_trucksim_39d 前 6 列):")
        print("  对应: Xpos, Ypos, Yaw, Vx, Vy, YawRate  -> 即 [x, y, yaw, vx, vy, wz]")
        print(f"  shape = {V.shape}")
        print(f"  前 3 行 (前6列):\n{V[:3, :6]}")
    return data


def summary_and_mapping():
    """格式对比与转换要点。"""
    print("\n" + "=" * 60)
    print("3. 格式对比与转换要点")
    print("=" * 60)
    print("""
参考轨迹 .mat（目标格式）:
  - 变量名: ref_trajectory
  - ref_trajectory.time: (N,) 时间向量
  - ref_trajectory.position: struct { x, y, yaw }，各 (N,) 或 (N,1)
  - ref_trajectory.velocity: struct { vx, vy, wz }，各 (N,) 或 (N,1)
  - MPC 内部使用: N×6 数组 [x, y, yaw, vx, vy, wz]

训练集 .mat（当前格式）:
  - 顶层变量通常包含: Delta_act, T_motor, Vehicle_state_trucksim_39d 等
  - 轨迹状态在 Vehicle_state_trucksim_39d 的前 6 列:
     列0: Xpos -> x
     列1: Ypos -> y
     列2: Yaw  -> yaw
     列3: Vx   -> vx
     列4: Vy   -> vy
     列5: YawRate -> wz
  - 时间: 若未单独给出，需根据采样间隔构造 (如 0.01s)

转换步骤（后续 convert_dataset_to_ref 可做）:
  1. 从训练集读取 Vehicle_state_trucksim_39d，取前 6 列 -> [x, y, yaw, vx, vy, wz]
  2. 构造 time 向量（按采样间隔与长度；sim_params.fixedStep 为仿真步长，若数据已降采样则用 0.01）
  3. 按 ref_trajectory 结构体组织 position / velocity
  4. 用 scipy.io.savemat 保存为 ref_trajectory 格式（注意 MATLAB 结构体用 dtype 或嵌套 dict）

实测摘要:
  - 参考轨迹: 顶层 ref_trajectory，squeeze_me=True 下为 0 维结构化数组，
    time/position/velocity 各字段；解析后 N×6，N=2501，dt=0.01s。
  - 训练集: 顶层 Delta_act(2501,6), T_motor(2501,6), Vehicle_state_trucksim_39d(2501,39)，
    前 6 列即 [Xpos,Ypos,Yaw,Vx,Vy,YawRate]；sim_params 含 time/fixedStep。
""")


def main():
    print("MAT 格式分析：参考轨迹 vs 训练集")
    print("参考轨迹文件:", REF_MAT)
    print("训练集文件:  ", DATA_MAT)
    analyze_ref_mat(REF_MAT)
    analyze_dataset_mat(DATA_MAT)
    summary_and_mapping()
    print("\n分析完成。")


if __name__ == "__main__":
    main()
