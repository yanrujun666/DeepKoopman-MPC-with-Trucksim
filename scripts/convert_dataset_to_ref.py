"""
将训练集 .mat 转为 MPC 参考轨迹 .mat 格式。

训练集与参考轨迹的采样间隔均为 0.01s，本脚本生成的时间向量也按 0.01s 间隔。

输入：训练集格式，需包含 Vehicle_state_trucksim_39d（前 6 列为 x, y, yaw, vx, vy, wz）。
输出：ref_trajectory 格式，与 generate_straight_acceleration_trajectory.m 生成的一致，
      且 x 已平移到以第一帧为原点（所有 x 减去第一帧的 x）。

用法:
  单文件:
    python scripts/convert_dataset_to_ref.py <输入.mat> <输出.mat>
    python scripts/convert_dataset_to_ref.py -i <输入.mat> -o <输出.mat>
  批量（data/all -> data/ref_traj/all，输出文件名加 _ref）:
    python scripts/convert_dataset_to_ref.py --batch
"""

import argparse
import os
import sys
import glob
import numpy as np
import scipy.io as scio

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 批量模式默认路径
DEFAULT_INPUT_DIR = os.path.join(ROOT, "data", "all")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "ref_traj", "all")

# 训练集中轨迹状态矩阵的变量名与所需列
STATE_KEY = "Vehicle_state_trucksim_39d"
STATE_COLS = 6  # x, y, yaw, vx, vy, wz
# 训练集与参考轨迹的采样间隔一致，均为 0.01s（100 Hz）
DT_S = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description="将训练集 .mat 转为 MPC 参考轨迹 .mat（x 以第一帧为原点）"
    )
    parser.add_argument(
        "input_mat",
        nargs="?",
        default=None,
        help="输入训练集 .mat 路径（单文件模式）",
    )
    parser.add_argument(
        "output_mat",
        nargs="?",
        default=None,
        help="输出参考轨迹 .mat 路径（单文件模式）",
    )
    parser.add_argument("-i", "--input", dest="input_flag", help="输入 .mat 路径")
    parser.add_argument("-o", "--output", dest="output_flag", help="输出 .mat 路径")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量转换：将 data/all 下所有 .mat 转为 data/ref_traj/all 下的 *_ref.mat",
    )
    args = parser.parse_args()
    if args.batch:
        return None, None, True
    input_path = args.input_flag or args.input_mat
    output_path = args.output_flag or args.output_mat
    if not input_path or not output_path:
        parser.print_help()
        sys.exit(1)
    return input_path, output_path, False


def check_input_format(data):
    """检查是否为合法训练集格式，返回 (Vehicle_state_trucksim_39d, 错误信息)。"""
    if STATE_KEY not in data:
        return None, f"缺少 '{STATE_KEY}'，当前顶层键: {[k for k in data.keys() if not k.startswith('__')]}"
    V = data[STATE_KEY]
    V = np.asarray(V)
    if V.ndim != 2:
        return None, f"'{STATE_KEY}' 应为二维数组，当前 ndim={V.ndim}"
    if V.shape[1] < STATE_COLS:
        return None, f"'{STATE_KEY}' 列数至少为 {STATE_COLS}，当前 shape={V.shape}"
    if V.shape[0] < 1:
        return None, f"'{STATE_KEY}' 行数至少为 1，当前 shape={V.shape}"
    return V, None


def build_time_vector(n):
    """构造时间向量，采样间隔 0.01s，与训练集及 ref 一致。"""
    return np.arange(n, dtype=float) * DT_S


def dataset_to_ref_arrays(V):
    """
    从 Vehicle_state_trucksim_39d 前 6 列得到 [x,y,yaw,vx,vy,wz]，
    y, yaw, vx, vy, wz 保持不变，x 减去第一帧的 x。
    """
    x = np.asarray(V[:, 0], dtype=float).flatten()
    y = np.asarray(V[:, 1], dtype=float).flatten()
    yaw = np.asarray(V[:, 2], dtype=float).flatten()
    vx = np.asarray(V[:, 3], dtype=float).flatten()
    vy = np.asarray(V[:, 4], dtype=float).flatten()
    wz = np.asarray(V[:, 5], dtype=float).flatten()
    # x 以第一帧为原点
    x0 = float(x[0])
    x = x - x0
    return x, y, yaw, vx, vy, wz


def build_ref_trajectory_struct(time, x, y, yaw, vx, vy, wz):
    """
    构建与 MATLAB ref_trajectory 兼容的结构体（numpy 结构化数组），
    使 loadmat(..., squeeze_me=True) 能正确解析。
    """
    n = len(time)
    # 与 generate_straight_acceleration_trajectory.m 一致：列向量 (N,1)
    t = np.asarray(time, dtype=float).reshape(-1, 1)
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    yaw = np.asarray(yaw, dtype=float).reshape(-1, 1)
    vx = np.asarray(vx, dtype=float).reshape(-1, 1)
    vy = np.asarray(vy, dtype=float).reshape(-1, 1)
    wz = np.asarray(wz, dtype=float).reshape(-1, 1)

    # 用「一行多列」构造：每行是 (field1, field2, ...)，不能是 ((row,),) 否则变成 1 赋给 3 字段
    position = np.array(
        [(x, y, yaw)], dtype=[("x", object), ("y", object), ("yaw", object)]
    )
    velocity = np.array(
        [(vx, vy, wz)], dtype=[("vx", object), ("vy", object), ("wz", object)]
    )
    ref_trajectory = np.array(
        [(t, position[0], velocity[0])],
        dtype=[("time", object), ("position", object), ("velocity", object)],
    )
    return ref_trajectory[0]


def convert(input_path, output_path):
    """执行转换并做格式检查。"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    data = scio.loadmat(input_path, simplify_cells=True)
    V, err = check_input_format(data)
    if err:
        raise ValueError(f"输入格式检查失败: {err}")

    n = V.shape[0]
    time = build_time_vector(n)

    x, y, yaw, vx, vy, wz = dataset_to_ref_arrays(V)
    ref_trajectory = build_ref_trajectory_struct(time, x, y, yaw, vx, vy, wz)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    scio.savemat(output_path, {"ref_trajectory": ref_trajectory}, format="5", do_compression=False)
    return n, float(x[0]), float(V[0, 0])


def run_batch(input_dir=None, output_dir=None):
    """将 input_dir 下所有 .mat 转为 output_dir 下的 *_ref.mat。"""
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_dir, "*.mat")
    mat_files = sorted(glob.glob(pattern))
    if not mat_files:
        print(f"未找到 .mat 文件: {pattern}")
        return
    print(f"批量转换: {input_dir} -> {output_dir}，共 {len(mat_files)} 个文件")
    ok, fail = 0, 0
    for inp in mat_files:
        basename = os.path.basename(inp)
        stem, ext = os.path.splitext(basename)
        out_name = stem + "_ref" + ext
        out_path = os.path.join(output_dir, out_name)
        try:
            convert(inp, out_path)
            print(f"  OK: {basename} -> {out_name}")
            ok += 1
        except Exception as e:
            print(f"  失败: {basename} - {e}", file=sys.stderr)
            fail += 1
    print(f"完成: 成功 {ok}, 失败 {fail}")


def main():
    input_path, output_path, batch = parse_args()
    try:
        if batch:
            run_batch()
        else:
            n, x0_new, x0_orig = convert(input_path, output_path)
            print(f"转换完成: {input_path} -> {output_path}")
            print(f"  轨迹点数: {n}")
            print(f"  原第一帧 x: {x0_orig:.6f} -> 平移后第一帧 x: {x0_new:.6f}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
