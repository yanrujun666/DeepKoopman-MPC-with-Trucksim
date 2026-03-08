"""
简易线性 MPC：基于学习到的 A、B 转移矩阵跟踪直线加速轨迹。
与 MPC 文件夹下已有代码完全隔离，仅用于验证 AB 模型。

状态约定（A/B 与 ref 一致）:
  (x, y): 固定在 t=0 的「初始车头系」下的位置（原点在起点，x 沿初始车头方向）
  (vx, vy): 随车动的自车系（车身系）下的纵向/横向速度
  yaw, yaw_rate: 世界系下航向角及角速度

动力学（列向量）: X_{t+1} = A.T @ X_t + B.T @ u_t
  X_t: 6x1 [x, y, yaw, vx, vy, yaw_rate]
  u_t: 12x1 [左前转矩, ..., 左前转角, ...] (转角单位: rad)
"""
from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from typing import Tuple, List

# 参数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "LinearModel-multiset.pth")

Np = 50
Nc = 40
dt = 0.01

# 约束：转矩 N·m，转角 rad
TORQUE_MIN = -1000.0
TORQUE_MAX = 1000.0
STEER_MIN = -30.0 * np.pi / 180.0   # rad
STEER_MAX = 30.0 * np.pi / 180.0

# 成本权重（状态误差 Q，控制 R）

# [x, y, yaw, vx, vy, yaw_rate] 位置/航向略大，速度适中
# Q_diag = np.array([80.0, 80.0, 40.0, 20.0, 20.0, 15.0])

# [x, y, yaw, vx, vy, yaw_rate] — x, y, yaw 为主要跟踪目标
Q_diag = np.array([10.0, 10.0, 10.0, 100.0, 10.0, 10.0])
# 控制简化为 2 维: [转矩标量, 转角标量]，W 将 2 维映射回 12 维
# 列1: 转矩均分到 6 轮; 列2: 前+1 中0 后-1 转角
W = np.array([
    [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
    [0, 1], [0, 1], [0, 0], [0, 0], [0, -1], [0, -1],
]).astype(float)   # 12x2
R_diag = np.array([1, 10])   # [转矩, 转角] 2 维
Nu_opt = 2

PRINT_INTERVAL = 100


def _print_mpc_matrix_properties(Phi: np.ndarray, Gamma: np.ndarray, H: np.ndarray) -> None:
    """Output numerical properties of MPC matrices (Phi, Gamma/THETA, H): cond, norm, rank, etc."""
    print("========== MPC matrix properties ==========")
    for name, M in [("Phi", Phi), ("Gamma (THETA)", Gamma), ("H", H)]:
        sh = M.shape
        c = np.linalg.cond(M) if M.size > 0 else float("nan")
        n2 = np.linalg.norm(M, 2)
        nf = np.linalg.norm(M, "fro")
        rk = np.linalg.matrix_rank(M, tol=1e-10)
        print(f"  {name}: shape={sh}, cond(2)={c:.6e}, norm2={n2:.6e}, normF={nf:.6e}, rank={rk}")
    try:
        eigs = np.linalg.eigvalsh(H)
        print(f"  H: min_eig={eigs.min():.6e}, max_eig={eigs.max():.6e}")
    except Exception:
        pass
    print("===========================================")


def _visualize_ab(A_dyn: np.ndarray, B_dyn: np.ndarray) -> None:
    """Visualize A_dyn (6x6) and B_dyn (6x12) as heatmaps.
    B_dyn[i,j] = effect of control j on state i (next); rows=states, cols=controls.
    """
    import matplotlib.pyplot as plt

    state_labels = ["x", "y", "yaw", "vx", "vy", "yaw_rate"]
    # 12 controls: 6 torques (FL,FR,ML,MR,RL,RR) + 6 steers
    u_labels = ["T_FL", "T_FR", "T_ML", "T_MR", "T_RL", "T_RR", "S_FL", "S_FR", "S_ML", "S_MR", "S_RL", "S_RR"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    im_a = ax_a.imshow(A_dyn, cmap="RdBu_r", aspect="equal", origin="lower")
    ax_a.set_xticks(range(6))
    ax_a.set_yticks(range(6))
    ax_a.set_xticklabels(state_labels)
    ax_a.set_yticklabels(state_labels)
    ax_a.set_xlabel("State (current)")
    ax_a.set_ylabel("State (next)")
    ax_a.set_title("A_dyn (6×6)")
    plt.colorbar(im_a, ax=ax_a, shrink=0.8)

    # B_dyn: shape (6, 12), row i = state i, col j = control j
    im_b = ax_b.imshow(B_dyn, cmap="RdBu_r", aspect="auto", origin="lower")
    ax_b.set_xticks(range(12))
    ax_b.set_yticks(range(6))
    ax_b.set_xticklabels(u_labels, rotation=45, ha="right")
    ax_b.set_yticklabels(state_labels)
    ax_b.set_xlabel("Control input (torque T_*, steer S_*)")
    ax_b.set_ylabel("State (next)")
    ax_b.set_title("B_dyn (6×12): row=state, col=control")
    plt.colorbar(im_b, ax=ax_b, shrink=0.8)

    plt.tight_layout()
    plt.show()


def load_ab(weights_path: str, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """从 .pth 加载 A、B，返回列向量形式的 A_dyn(6x6), B_dyn(6x12)。"""
    import torch
    w = torch.load(weights_path, map_location="cpu")
    dkm = w["dkm"]
    A = dkm["A"].numpy()   # (6,6)
    B = dkm["B"].numpy()   # (12,6)
    A_dyn = A.T            # 6x6
    B_dyn = B.T            # 6x12
    if visualize:
        _visualize_ab(A_dyn, B_dyn)
    return A_dyn, B_dyn


def build_prediction_matrices(
    A_dyn: np.ndarray,
    B_eff: np.ndarray,
    Np: int,
    Nc: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建 X = Phi @ x0 + Gamma @ U。U = [u_0; ...; u_{Nc-1}]，u_k 为 2 维 (Nu_opt*Nc x 1)。"""
    n_x, n_u = B_eff.shape[0], B_eff.shape[1]
    Phi = np.zeros((Np * n_x, n_x))
    Gamma = np.zeros((Np * n_x, Nc * n_u))

    for k in range(1, Np + 1):
        Phi[6 * (k - 1) : 6 * k, :] = np.linalg.matrix_power(A_dyn, k)

    for j in range(Nc):
        for k in range(j + 1, Np + 1):
            if j < Nc - 1:
                coef = np.linalg.matrix_power(A_dyn, k - 1 - j) @ B_eff
            else:
                s = np.zeros((n_x, n_x))
                for i in range(k - Nc + 1):
                    s += np.linalg.matrix_power(A_dyn, i)
                coef = s @ B_eff
            Gamma[6 * (k - 1) : 6 * k, n_u * j : n_u * (j + 1)] = coef

    return Phi, Gamma


def generate_straight_accel_ref(
    t_end: float,
    dt: float,
    v0: float = 0.0,
    ax: float = 1.0,
) -> np.ndarray:
    """直线加速参考: x = v0*t + 0.5*ax*t^2, y=0, yaw=0, vx=v0+ax*t, vy=0, yaw_rate=0。"""
    n_steps = int(round(t_end / dt)) + 1
    ref = np.zeros((n_steps, 6))
    for i in range(n_steps):
        t = i * dt
        ref[i, 0] = v0 * t + 0.5 * ax * t * t
        ref[i, 1] = 0.0
        ref[i, 2] = 0.0
        ref[i, 3] = v0 + ax * t
        ref[i, 4] = 0.0
        ref[i, 5] = 0.0
    return ref


def generate_const_vel_ref(
    t_end: float,
    dt: float,
    v0: float = 5.0,
) -> np.ndarray:
    """Constant velocity reference from [0,0,0,v0,0,0]: x=v0*t, y=0, yaw=0, vx=v0, vy=0, yaw_rate=0."""
    n_steps = int(round(t_end / dt)) + 1
    ref = np.zeros((n_steps, 6))
    for i in range(n_steps):
        t = i * dt
        ref[i, 0] = v0 * t
        ref[i, 1] = 0.0
        ref[i, 2] = 0.0
        ref[i, 3] = v0
        ref[i, 4] = 0.0
        ref[i, 5] = 0.0
    return ref


def run_naive_mpc():
    A_dyn, B_dyn = load_ab(WEIGHTS_PATH)
    B_eff = B_dyn @ W   # 6x2
    Phi, Gamma = build_prediction_matrices(A_dyn, B_eff, Np, Nc)

    Q_bar = np.kron(np.eye(Np), np.diag(Q_diag))
    R_bar = np.kron(np.eye(Nc), np.diag(R_diag))
    H = 2 * (Gamma.T @ Q_bar @ Gamma + R_bar)
    H = (H + H.T) / 2
    H += 1e-8 * np.eye(H.shape[0])

    _print_mpc_matrix_properties(Phi, Gamma, H)

    lb_full = np.array([TORQUE_MIN, STEER_MIN])
    ub_full = np.array([TORQUE_MAX, STEER_MAX])
    lb = np.tile(lb_full, Nc)
    ub = np.tile(ub_full, Nc)
    n_u = Nu_opt * Nc
    C_T = np.hstack([np.eye(n_u), -np.eye(n_u)])
    b_box = np.concatenate([lb, -ub])

    ref_traj = generate_straight_accel_ref(t_end=10.0, dt=dt, v0=5.0, ax=1.0)
    # ref_traj = generate_const_vel_ref(t_end=10.0, dt=dt, v0=5.0)
    n_sim = min(ref_traj.shape[0] - 1, 500)

    x = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    x_hist: List[np.ndarray] = []
    u_hist: List[np.ndarray] = []

    try:
        import quadprog
    except ImportError:
        print("请安装 quadprog: pip install quadprog")
        return

    for step in range(n_sim):
        x_hist.append(x.copy())
        idx = min(step + Np, ref_traj.shape[0])
        ref_block = ref_traj[step : idx]
        if ref_block.shape[0] < Np:
            pad = np.tile(ref_traj[-1], (Np - ref_block.shape[0], 1))
            ref_block = np.vstack([ref_block, pad])
        else:
            ref_block = ref_block[:Np]
        Ref = ref_block.ravel(order="C")

        g = 2 * Gamma.T @ Q_bar @ (Phi @ x - Ref)
        try:
            result = quadprog.solve_qp(H, g, C_T, b_box, meq=0)
            U_opt = np.asarray(result[0]).ravel()
        except Exception as e:
            print(f"step {step} QP 失败: {e}")
            U_opt = np.zeros(Nu_opt * Nc)

        u0_opt = U_opt[:Nu_opt]
        u0_full = (W @ u0_opt.reshape(-1, 1)).ravel()
        u_hist.append(u0_full.copy())
        x = (A_dyn @ x.reshape(-1, 1) + B_eff @ u0_opt.reshape(-1, 1)).ravel()

        if step % PRINT_INTERVAL == 0:
            ref_cur = ref_traj[min(step, ref_traj.shape[0] - 1)]
            print(f"--- step {step} ---")
            print("当前状态 X [x,y,yaw,vx,vy,yaw_rate]:", np.round(x, 6))
            print("参考状态 R [x,y,yaw,vx,vy,yaw_rate]:", np.round(ref_cur, 6))
            print("控制 u_opt [转矩(N·m), 转角(rad)]:", np.round(u0_opt, 6))
            print("控制 u_full (12维):", np.round(u0_full, 6))
            print()

    x_hist.append(x.copy())
    _plot_results(x_hist, u_hist, ref_traj, n_sim)
    print("仿真结束。")


def _plot_results(
    x_hist: List[np.ndarray],
    u_hist: List[np.ndarray],
    ref_traj: np.ndarray,
    n_sim: int,
) -> None:
    """绘制跟踪效果与控制信号。"""
    import matplotlib.pyplot as plt

    t_state = np.arange(n_sim + 1) * dt
    t_ctrl = np.arange(n_sim) * dt
    X = np.array(x_hist)
    R = ref_traj[: n_sim + 1]
    U = np.array(u_hist)

    # x-y trajectory
    fig0, ax0 = plt.subplots(1, 1, figsize=(6, 5))
    ax0.plot(X[:, 0], X[:, 1], label="Actual", color="C0")
    ax0.plot(R[:, 0], R[:, 1], label="Reference", color="C1", linestyle="--")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("Trajectory (x-y)")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)
    ax0.axis("equal")
    plt.tight_layout()

    state_names = ["x (m)", "y (m)", "yaw (rad)", "vx (m/s)", "vy (m/s)", "yaw_rate (rad/s)"]
    fig1, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i in range(6):
        axes[i].plot(t_state, X[:, i], label="Actual", color="C0")
        axes[i].plot(t_state, R[:, i], label="Reference", color="C1", linestyle="--")
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-2].set_xlabel("t (s)")
    axes[-1].set_xlabel("t (s)")
    fig1.suptitle("State Tracking")
    plt.tight_layout()

    fig2, (ax_tq, ax_st) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    torque_labels = ["FL", "FR", "ML", "MR", "RL", "RR"]
    for i in range(6):
        ax_tq.plot(t_ctrl, U[:, i], label=torque_labels[i], alpha=0.8)
    ax_tq.set_ylabel("Torque (N·m)")
    ax_tq.legend(loc="upper right", ncol=3, fontsize=8)
    ax_tq.grid(True, alpha=0.3)
    steer_labels = ["FL", "FR", "ML", "MR", "RL", "RR"]
    for i in range(6):
        ax_st.plot(t_ctrl, np.rad2deg(U[:, 6 + i]), label=steer_labels[i], alpha=0.8)
    ax_st.set_ylabel("Steer angle (°)")
    ax_st.set_xlabel("t (s)")
    ax_st.legend(loc="upper right", ncol=3, fontsize=8)
    ax_st.grid(True, alpha=0.3)
    fig2.suptitle("Control Inputs")
    plt.tight_layout()

    plt.show()


def simulate_constant_control_and_plot() -> None:
    """
    Simulate with constant control. State convention: (x,y) in initial frame (fixed at t=0),
    (vx, vy) in body frame. x-y plot uses (x, y) directly (no conversion).
    """
    import matplotlib.pyplot as plt

    A_dyn, B_dyn = load_ab(WEIGHTS_PATH)
    x0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    u_const = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -0.5, -0.5])
    t_end = 1
    n_steps = int(round(t_end / dt))

    x_hist: List[np.ndarray] = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = (A_dyn @ x.reshape(-1, 1) + B_dyn @ u_const.reshape(-1, 1)).ravel()
        x_hist.append(x.copy())

    X = np.array(x_hist)
    t_axis = np.arange(n_steps + 1) * dt

    # (x, y) is already in initial vehicle frame (fixed at t=0), plot directly
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    ax1.plot(X[:, 0], X[:, 1], color="C0", label="Trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Trajectory (x-y) under constant control (initial frame)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    plt.tight_layout()

    state_names = [
        "x (m, init frame)",
        "y (m, init frame)",
        "yaw (rad)",
        "vx (m/s, body)",
        "vy (m/s, body)",
        "yaw_rate (rad/s)",
    ]
    fig2, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i in range(6):
        axes[i].plot(t_axis, X[:, i], color="C0")
        axes[i].set_ylabel(state_names[i])
        axes[i].set_xlabel("t (s)")
        axes[i].grid(True, alpha=0.3)
    fig2.suptitle("State evolution under constant control")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Visualize A, B matrices (uncomment to run):
    # load_ab(WEIGHTS_PATH, visualize=True)
    # run_naive_mpc()
    simulate_constant_control_and_plot()
