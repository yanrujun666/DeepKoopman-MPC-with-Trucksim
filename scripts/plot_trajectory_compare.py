"""
仿真结束后：读取仿真过程中记录的实车 xy 轨迹 CSV，与参考轨迹 .mat 的 xy 对比绘图。

仿真时 MPC 端会将每步的 (step, x, y) 追加到 data/vehicle_trajectory_log.csv；
本脚本读取该 CSV 与（可选）参考轨迹 .mat，绘制 x-y 平面上的对比图。

用法:
  # 使用默认 CSV 与默认参考轨迹绘图
  python scripts/plot_trajectory_compare.py

  # 指定 CSV 或参考 .mat
  python scripts/plot_trajectory_compare.py --csv data/exp_traj_log/vehicle_trajectory_log.csv --ref data/ref_traj/all/xxx_ref.mat
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "exp_traj_log", "vehicle_trajectory_log.csv")
DEFAULT_REF = os.path.join(ROOT, "data", "ref_traj", "all", "all_wheel_steer_Scenario_snake_acc_5m_s_ref.mat")


def load_vehicle_csv(csv_path: str):
    """加载仿真记录的实车轨迹 CSV：step,x,y -> (steps, x_arr, y_arr)。"""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    steps = data[:, 0].astype(int)
    x = data[:, 1]
    y = data[:, 2]
    return steps, x, y


def _flat_first(a):
    """取 0-d 或 1-d 数组的第一个元素（用于 MATLAB 嵌套 struct）。"""
    a = np.atleast_1d(a)
    while a.size == 1 and a.ndim >= 1 and a.dtype.names:
        a = a.flat[0]
        a = np.atleast_1d(a)
    return a.flatten()


def load_ref_xy_from_mat(mat_path: str):
    """
    从参考轨迹 .mat 中解析出 x, y 数组。
    支持：ref_trajectory 结构体（position.x / position.y）、或 'position' N×3、或 'Pos' N×3。
    """
    import scipy.io as scio
    ref_data = scio.loadmat(mat_path, squeeze_me=True)

    if "ref_trajectory" in ref_data:
        rt = ref_data["ref_trajectory"]
        # 兼容 numpy 结构体：ref_trajectory.position.x / .y（可能为 0-d 或 1-d）
        try:
            rt_flat = rt.flat[0] if rt.size > 0 else rt[()]
            pos = rt_flat["position"] if hasattr(rt_flat, "dtype") and getattr(rt_flat.dtype, "names", None) and "position" in rt_flat.dtype.names else rt_flat
            pos_flat = pos.flat[0] if isinstance(pos, np.ndarray) and pos.size > 0 and (pos.shape == () or pos.ndim >= 1) else pos
            for xf, yf in [("x", "y"), ("X", "Y")]:
                names = getattr(pos_flat.dtype, "names", None) if hasattr(pos_flat, "dtype") else None
                if names and xf in names and yf in names:
                    return _flat_first(pos_flat[xf]), _flat_first(pos_flat[yf])
        except Exception:
            pass
    if "position" in ref_data:
        pos = np.array(ref_data["position"])
        if pos.ndim == 1:
            pos = pos.reshape(-1, 3)
        return pos[:, 0], pos[:, 1]
    if "Pos" in ref_data:
        pos = np.array(ref_data["Pos"])
        if pos.ndim == 1:
            pos = pos.reshape(-1, 3)
        return pos[:, 0], pos[:, 1]
    raise KeyError("mat 中未找到 ref_trajectory / position / Pos，无法解析参考轨迹 xy")


def main():
    parser = argparse.ArgumentParser(
        description="绘制实车 xy 轨迹与参考轨迹对比图（仿真结束后运行）"
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"实车轨迹 CSV 路径，默认 {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help=f"参考轨迹 .mat 路径，默认 {DEFAULT_REF}",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出图片路径（可选）；不指定则弹窗显示",
    )
    args = parser.parse_args()
    csv_path = os.path.normpath(os.path.join(ROOT, args.csv)) if not os.path.isabs(args.csv) else args.csv

    if not os.path.isfile(csv_path):
        print(f"错误：未找到实车轨迹 CSV: {csv_path}")
        print("请先运行 Simulink 仿真，仿真过程中会自动写入该文件。")
        sys.exit(1)

    steps, x_act, y_act = load_vehicle_csv(csv_path)
    print(f"已加载实车轨迹: {len(x_act)} 点，来自 {csv_path}")

    ref_x, ref_y = None, None
    ref_path = os.path.normpath(os.path.join(ROOT, args.ref)) if args.ref and not os.path.isabs(args.ref) else (args.ref or "")
    if ref_path and os.path.isfile(ref_path):
        try:
            ref_x, ref_y = load_ref_xy_from_mat(ref_path)
            print(f"已加载参考轨迹: {len(ref_x)} 点，来自 {ref_path}")
        except Exception as e:
            print(f"警告：加载参考轨迹失败 ({e})，仅绘制实车轨迹。")
            ref_x, ref_y = None, None
    elif ref_path:
        print(f"警告：未找到参考轨迹文件 {ref_path}，仅绘制实车轨迹。")

    try:
        import matplotlib
        # 仅保存到文件时使用非交互后端；未指定 --out 时用默认后端以便 plt.show() 弹窗
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装 matplotlib: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x_act, y_act, "b-", linewidth=1.5, label="Vehicle (x, y)")
    if ref_x is not None and ref_y is not None:
        ax.plot(ref_x, ref_y, "r--", linewidth=1.0, alpha=0.8, label="Reference (x, y)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vehicle vs Reference Trajectory (x-y)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    if args.out:
        out_path = os.path.normpath(os.path.join(ROOT, args.out)) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
