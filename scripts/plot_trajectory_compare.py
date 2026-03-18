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


def align_ref_to_time(ref_x: np.ndarray, ref_y: np.ndarray, t: np.ndarray, dt_ref: float):
    """
    将参考轨迹按时间对齐到实车采样点。

    - 实车时间轴：t（秒）
    - 参考时间轴：t_ref = [0, dt_ref, 2*dt_ref, ...]
    - 若 t 超出参考时间范围：使用首/末值钳位外推
    """
    ref_x = np.asarray(ref_x).reshape(-1)
    ref_y = np.asarray(ref_y).reshape(-1)
    t = np.asarray(t).astype(float).reshape(-1)

    if ref_x.size == 0 or ref_y.size == 0:
        raise ValueError("参考轨迹为空，无法对齐")
    if ref_x.size != ref_y.size:
        raise ValueError(f"参考轨迹长度不一致: len(x)={ref_x.size}, len(y)={ref_y.size}")
    if dt_ref <= 0:
        raise ValueError(f"dt_ref 必须为正数，当前 dt_ref={dt_ref}")

    t_ref = np.arange(ref_x.size, dtype=float) * float(dt_ref)
    x_ref = np.interp(t, t_ref, ref_x, left=ref_x[0], right=ref_x[-1])
    y_ref = np.interp(t, t_ref, ref_y, left=ref_y[0], right=ref_y[-1])
    return x_ref, y_ref


def compute_error_metrics(e: np.ndarray):
    e = np.asarray(e).reshape(-1)
    if e.size == 0:
        raise ValueError("误差序列为空")
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))
    max_abs = float(np.max(np.abs(e)))
    mean = float(np.mean(e))
    std = float(np.std(e))
    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "mean": mean,
        "std": std,
        "n": int(e.size),
    }


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
    parser.add_argument(
        "--dt_csv",
        type=float,
        default=0.001,
        help="CSV 每个 step 的时间间隔（秒）。默认 0.001",
    )
    parser.add_argument(
        "--dt_ref",
        type=float,
        default=0.01,
        help="参考轨迹每个 index 的时间间隔（秒）。默认 0.01",
    )
    parser.add_argument(
        "--save_metrics",
        default=None,
        help="将误差指标保存到该路径（.txt 或 .csv）；不指定则仅打印",
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

    # ====== 误差指标与误差-时间曲线 ======
    if ref_x is not None and ref_y is not None:
        # CSV: step 间隔 dt_csv；参考 mat: index 间隔 dt_ref
        t = (steps - steps[0]) * float(args.dt_csv)
        x_ref, y_ref = align_ref_to_time(ref_x, ref_y, t, dt_ref=float(args.dt_ref))
        ex = x_act - x_ref
        ey = y_act - y_ref
        e2d = np.sqrt(ex**2 + ey**2)

        mx = compute_error_metrics(ex)
        my = compute_error_metrics(ey)
        m2d = compute_error_metrics(e2d)

        def _fmt(m):
            return f"rmse={m['rmse']:.6g}, mae={m['mae']:.6g}, max_abs={m['max_abs']:.6g}, mean={m['mean']:.6g}, std={m['std']:.6g}, n={m['n']}"

        print("\n===== 轨迹跟踪误差指标 =====")
        print(f"x误差:  {_fmt(mx)}")
        print(f"y误差:  {_fmt(my)}")
        print(f"2D误差: {_fmt(m2d)} (sqrt(ex^2+ey^2))")

        # 保存指标（可选）
        if args.save_metrics:
            metrics_path = os.path.normpath(os.path.join(ROOT, args.save_metrics)) if not os.path.isabs(args.save_metrics) else args.save_metrics
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
            ext = os.path.splitext(metrics_path)[1].lower()
            if ext == ".csv":
                import csv

                rows = [
                    ("ex",) + tuple(mx[k] for k in ["rmse", "mae", "max_abs", "mean", "std", "n"]),
                    ("ey",) + tuple(my[k] for k in ["rmse", "mae", "max_abs", "mean", "std", "n"]),
                    ("e2d",) + tuple(m2d[k] for k in ["rmse", "mae", "max_abs", "mean", "std", "n"]),
                ]
                with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["signal", "rmse", "mae", "max_abs", "mean", "std", "n"])
                    w.writerows(rows)
            else:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write("Trajectory tracking error metrics\n")
                    f.write(f"csv={csv_path}\n")
                    if ref_path:
                        f.write(f"ref={ref_path}\n")
                    f.write(f"dt_csv={args.dt_csv}\n")
                    f.write(f"dt_ref={args.dt_ref}\n\n")
                    f.write(f"x_error:  {_fmt(mx)}\n")
                    f.write(f"y_error:  {_fmt(my)}\n")
                    f.write(f"2d_error: {_fmt(m2d)}\n")
            print(f"已保存误差指标: {metrics_path}")

        # 误差随时间曲线（x/y）
        fig_e, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        abs_ex = np.abs(ex)
        abs_ey = np.abs(ey)
        axs[0].plot(t, abs_ex, "k-", linewidth=1.0)
        axs[0].set_ylabel("|x error| (m)")
        axs[0].grid(True, alpha=0.3)
        axs[1].plot(t, abs_ey, "k-", linewidth=1.0)
        axs[1].set_ylabel("|y error| (m)")
        axs[1].set_xlabel("time (s)")
        axs[1].grid(True, alpha=0.3)
        fig_e.suptitle("Tracking Error vs Time")

        # 在各自子图中标注对应指标（RMSE/MAE/MaxAbs）
        axs[0].text(
            0.02,
            0.98,
            f"RMSE={mx['rmse']:.4g}m, MAE={mx['mae']:.4g}m, Max|e|={mx['max_abs']:.4g}m",
            transform=axs[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
        )
        axs[1].text(
            0.02,
            0.98,
            f"RMSE={my['rmse']:.4g}m, MAE={my['mae']:.4g}m, Max|e|={my['max_abs']:.4g}m",
            transform=axs[1].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
        )

        # 保存误差曲线图（如果 --out 指定则自动同目录保存一份）
        if args.out:
            out_path = os.path.normpath(os.path.join(ROOT, args.out)) if not os.path.isabs(args.out) else args.out
            out_dir = os.path.dirname(out_path) or "."
            base, _ = os.path.splitext(os.path.basename(out_path))
            err_path = os.path.join(out_dir, f"{base}_err_xy.png")
            os.makedirs(out_dir, exist_ok=True)
            fig_e.savefig(err_path, dpi=150, bbox_inches="tight")
            print(f"已保存误差曲线图: {err_path}")

    if args.out:
        out_path = os.path.normpath(os.path.join(ROOT, args.out)) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
