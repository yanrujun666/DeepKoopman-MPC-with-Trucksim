import os
import torch
import tqdm
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import ConcatDataset

from eval_sequence import (
    CustomEncoderUnscaledv2WithoutNorm,
    Koopmanv1,
    VehicleDynamicDataset5C,
    condition_dir_dict,
    DEVICE,
    CONTROL_MIN,
    CONTROL_MAX,
    count_parameters,
)


class KoopmanMPC(nn.Module):
    """
    使用 Koopman 线性动力学 z_{k+1} = z_k A + u_k B 的
    传统 QP 型 MPC 控制器（在 Koopman 模型上闭环仿真）。
    保持 12 维控制输入不变，但在代价函数中加入“软约束”，
    鼓励：各轮扭矩一致、前轮转角一致、中轮为 0、后轮与前轮相反。
    """

    def __init__(
        self,
        dkm: Koopmanv1,
        state_dim: int = 6,
        control_dim: int = 12,
        horizon: int = 10,
        q_weights=None,
        r_weights=None,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.dkm = dkm
        self.state_dim = state_dim
        self.control_dim = control_dim  # 仍为 12 维
        self.horizon = horizon
        self.device = device

        # 归一化后的 12 维控制约束：[0, 1]
        self.u_min = torch.zeros(self.control_dim, device=device)
        self.u_max = torch.ones(self.control_dim, device=device)

        # 转角相关的归一化常数：
        # 物理转角为 0 rad 时对应的归一化中心（理论上为 0.5）
        steer_min = CONTROL_MIN[6].item()
        steer_max = CONTROL_MAX[6].item()
        self.steer_center_norm = float((0.0 - steer_min) / (steer_max - steer_min))
        # 当前轮与后轮物理上相反（d_rear = -d_front）时，
        # 对应的归一化量之和的目标值（对称 ±pi/2 情况下理论上为 1.0）
        self.steer_opposite_sum_norm = float((-2.0 * steer_min) / (steer_max - steer_min))

        if q_weights is None:
            # 对前 6 维物理状态赋予较大权重，其余隐变量较小
            base = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
            tail = torch.ones(self.dkm.A.shape[0] - base.numel(), device=device)
            q_weights = torch.cat([base, tail], dim=0)
        
        if r_weights is None:
            r_weights = torch.zeros(self.control_dim, device=device)

        i_weights = torch.ones(self.control_dim, device=device) * 0.1

        self.register_buffer("q_weights", q_weights)
        self.register_buffer("r_weights", r_weights)
        self.register_buffer("i_weights", i_weights)

        # # 软约束权重
        # self.lambda_torque_equal = 100.0
        # self.lambda_middle_zero = 100.0
        # self.lambda_steer_relation = 100.0

        # 预先缓存为 numpy，方便传入 QP 求解器
        with torch.no_grad():
            A = self.dkm.A.detach().cpu().numpy()
            B = self.dkm.B.detach().cpu().numpy()
        self.A_np = A
        self.B_np = B

        self.Q_np = np.diag(self.q_weights.detach().cpu().numpy())
        self.R_np = np.diag(self.r_weights.detach().cpu().numpy())
        self.I_np = np.diag(self.i_weights.detach().cpu().numpy())
        self.u_min_np = self.u_min.detach().cpu().numpy()
        self.u_max_np = self.u_max.detach().cpu().numpy()

    def rollout(self, z0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        使用 Koopman 动力学在 Koopman 空间前向展开。
        z0:  [B, hidden_dim]
        U:   [B, H, control_dim]（已归一化）
        返回 z_traj: [B, H, hidden_dim]
        """
        B, H, _ = U.shape
        hidden_dim = z0.shape[-1]
        z = z0
        traj = []
        for k in range(H):
            u_k = U[:, k, :]  # [B, 12]
            z = z @ self.dkm.A + u_k @ self.dkm.B  # [B, hidden_dim]
            traj.append(z)
        traj = torch.stack(traj, dim=1)  # [B, H, hidden_dim]
        assert traj.shape == (B, H, hidden_dim)
        return traj

    def _solve_mpc_qp_single(self, z0_np: np.ndarray, ref_traj_np: np.ndarray, u_init_np: np.ndarray = None) -> np.ndarray:
        """
        使用 cvxpy 为单个样本求解有限时域 QP：
            min sum_k (x_k - x_ref_k)^T Q (x_k - x_ref_k) + u_k^T R u_k
            s.t. z_{k+1} = z_k A + u_k B
                 u_min <= u_k <= u_max
        其中 x_k 为 z_k 的前 state_dim 个分量。

        z0_np:       [hidden_dim]
        ref_traj_np: [H, state_dim]
        u_init_np:   [H, control_dim] 或 None（用于 warm start）
        返回 u_opt_np: [H, control_dim]
        """
        H = ref_traj_np.shape[0]
        n = z0_np.shape[0]
        m = self.control_dim

        A = self.A_np
        B = self.B_np
        Q = self.Q_np
        R = self.R_np
        I = self.I_np

        z = cp.Variable((H + 1, n))
        u = cp.Variable((H, m))

        constraints = [z[0, :] == z0_np]
        for k in range(H):
            # Koopman 动力学约束
            constraints.append(z[k + 1, :] == z[k, :] @ A + u[k, :] @ B)
            # 归一化控制范围约束
            constraints.append(u[k, :] >= self.u_min_np)
            constraints.append(u[k, :] <= self.u_max_np)

            # ------------- 硬约束：控制结构 -------------
            # 记：u = [T_FL, T_FR, T_ML, T_MR, T_RL, T_RR,
            #          delta_FL, delta_FR, delta_ML, delta_MR, delta_RL, delta_RR]
            T_FL, T_FR, T_ML, T_MR, T_RL, T_RR = (
                u[k, 0],
                u[k, 1],
                u[k, 2],
                u[k, 3],
                u[k, 4],
                u[k, 5],
            )
            d_FL, d_FR, d_ML, d_MR, d_RL, d_RR = (
                u[k, 6],
                u[k, 7],
                u[k, 8],
                u[k, 9],
                u[k, 10],
                u[k, 11],
            )

            # 1) 所有车轮转矩相同（归一化后相等）
            constraints += [
                T_FL == T_FR,
                T_FR == T_ML,
                T_ML == T_MR,
                T_MR == T_RL,
                T_RL == T_RR,
            ]

            # 4) 左右轮转角相同（每一轴）
            constraints += [
                d_FL == d_FR,  # 前轴左右一致
                d_ML == d_MR,  # 中轴左右一致
                d_RL == d_RR,  # 后轴左右一致
            ]

            # 3) 中轴两轮物理转角为 0：
            #    在归一化空间中，等价于 d_ML = d_MR = steer_center_norm
            constraints += [
                d_ML == self.steer_center_norm,
                d_MR == self.steer_center_norm,
            ]

            # 2) 前轴两轮与后轴两轮物理转角相反：
            #    归一化变量满足 d_R ≈ steer_opposite_sum_norm - d_F，
            #    即 d_R + d_F = steer_opposite_sum_norm
            constraints += [
                d_RL + d_FL == self.steer_opposite_sum_norm,
                d_RR + d_FR == self.steer_opposite_sum_norm,
            ]

        cost = 0
        for k in range(H):
            x_k = z[k + 1, :]
            r_k = ref_traj_np[k]
            # 状态跟踪代价
            cost += cp.quad_form(x_k - r_k, Q)

            # 基本控制能量代价
            # if u_init_np is not None:
            #     cost += cp.quad_form(u[k, :] - u_init_np[k, :], R)
        
            # cost += cp.quad_form(u[k, :] - 0.5, R)

            # # ---------- 软约束：鼓励期望的控制结构 ----------
            # # 记：u = [T_FL, T_FR, T_ML, T_MR, T_RL, T_RR,
            # #          delta_FL, delta_FR, delta_ML, delta_MR, delta_RL, delta_RR]
            # T_FL, T_FR, T_ML, T_MR, T_RL, T_RR = u[k, 0], u[k, 1], u[k, 2], u[k, 3], u[k, 4], u[k, 5]
            # d_FL, d_FR, d_ML, d_MR, d_RL, d_RR = u[k, 6], u[k, 7], u[k, 8], u[k, 9], u[k, 10], u[k, 11]

            # # 1) 6 个轮扭矩尽量相等：对相邻差分加惩罚
            # torque_equal_penalty = (
            #     cp.square(T_FL - T_FR)
            #     + cp.square(T_FR - T_ML)
            #     + cp.square(T_ML - T_MR)
            #     + cp.square(T_MR - T_RL)
            #     + cp.square(T_RL - T_RR)
            # )
            # cost += self.lambda_torque_equal * torque_equal_penalty

            # # 2) 中间两轮物理转角尽量为 0
            # #    在归一化空间中，对应于“接近 steer_center_norm”
            # middle_zero_penalty = (
            #     cp.square(d_ML - self.steer_center_norm)
            #     + cp.square(d_MR - self.steer_center_norm)
            # )
            # cost += self.lambda_middle_zero * middle_zero_penalty

            # # 3) 前轮一致、后轮一致，且后轮物理转角与前轮物理转角相反
            # #    对于归一化变量，d_FL/d_FR、d_RL/d_RR 仍要求一致；
            # #    “后轮 ≈ -前轮” 映射为 “后轮与前轮的归一化量之和 ≈ steer_opposite_sum_norm”
            # steer_rel_penalty = (
            #     cp.square(d_FL - d_FR)                         # 前轮一致
            #     + cp.square(d_RL - d_RR)                       # 后轮一致
            #     + cp.square(d_RL + d_FL - self.steer_opposite_sum_norm)
            #     + cp.square(d_RR + d_FR - self.steer_opposite_sum_norm)
            # )
            # cost += self.lambda_steer_relation * steer_rel_penalty

        for k in range(1, H):
            cost += cp.quad_form(u[k, :] - u[k-1, :], I)

        prob = cp.Problem(cp.Minimize(cost), constraints)

        if u_init_np is not None:
            u.value = u_init_np

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            prob.solve(warm_start=True, verbose=True)

        if u.value is None:
            # 求解失败时退化为零控制
            return np.ones((H, m), dtype=np.float32) * 0.5

        return np.asarray(u.value, dtype=np.float32)

    def optimize(self, z0: torch.Tensor, ref_traj_states: torch.Tensor, u_init: torch.Tensor = None) -> torch.Tensor:
        """
        传统 MPC（二次规划）求解：
        z0:              [B, hidden_dim]
        ref_traj_states: [B, H, state_dim]（状态参考轨迹，和 z 的坐标系一致）
        u_init:          [B, H, control_dim] 或 None（可选 warm start，已归一化）
        返回 U_opt:      [B, H, control_dim]（已在 [0,1] 范围内）
        """
        B, H, _ = ref_traj_states.shape
        assert B == 1, "当前实现仅支持 batch_size = 1 的 MPC 求解"

        z0_np = z0[0].detach().cpu().numpy()
        ref_np = ref_traj_states[0].detach().cpu().numpy()
        u_init_np = None
        if u_init is not None:
            u_init_np = u_init[0].detach().cpu().numpy()

        u_opt_np = self._solve_mpc_qp_single(z0_np, ref_np, u_init_np)
        U_opt = torch.from_numpy(u_opt_np).unsqueeze(0).to(self.device)  # [1, H, m]
        return U_opt


def build_model(ckpt_path: str):
    """
    加载与 eval_sequence.py 中一致的编码器与 Koopman 模型。
    """
    encoder = CustomEncoderUnscaledv2WithoutNorm(state_dim=6, hidden_dim=16, layer_depth=6)
    dkm = Koopmanv1(state_dim=6, hidden_dim=16, control_dim=12)

    print("Params of encoder:", count_parameters(encoder) / 1e6, "M")
    print("Params of dkm:", count_parameters(dkm) / 1e6, "M")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print("Loaded ckpt epoch:", ckpt.get("epoch", "NA"))
    print("Eval results in ckpt:", ckpt.get("eval_results", "NA"))

    encoder.load_state_dict(ckpt["encoder"])
    dkm.load_state_dict(ckpt["koopman"])

    encoder = encoder.to(DEVICE)
    dkm = dkm.to(DEVICE)
    encoder.eval()
    dkm.eval()
    return encoder, dkm


def prepare_valid_loader(eval_condition: str, eval_length: int, batch_size: int):
    eval_condition = eval_condition.lower()
    assert eval_condition in list(condition_dir_dict.keys()), "no such condition"

    if eval_length >= 300:
        skip = 25
    else:
        skip = 100

    valid_dir = condition_dir_dict[eval_condition]
    # valid_files = [os.path.join(valid_dir, x) for x in sorted(os.listdir(valid_dir))]
    valid_files = ['data/c1_all_wheel_steer/test/VehicleParams_IzuA_IzulA_RrA_CdA_BcdA/Scenario_snake_acc_5m_s.mat']
    valid_dsts = []
    for file in valid_files[:1]:
        valid_dsts.append(
            VehicleDynamicDataset5C(
                mat_fpath=file,
                horizon=eval_length,
                device=DEVICE,
                show_log=False,
                skip=skip,
            )
        )
    valid_dst = ConcatDataset(valid_dsts)
    valid_loader = torch.utils.data.DataLoader(
        valid_dst,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return valid_loader


def mpc_closed_loop_eval(
    encoder: CustomEncoderUnscaledv2WithoutNorm,
    dkm: Koopmanv1,
    data_loader,
    mpc_horizon: int = 10,
):
    """
    在测试集上进行基于 Koopman 动力学的 MPC 闭环轨迹跟踪评估。
    以数据集中的真实轨迹为参考轨迹，MPC 在 Koopman 模型上生成控制输入，
    形成闭环仿真，并统计轨迹跟踪误差。
    """

    # 误差统计
    traj_ade = 0.0
    traj_fde = 0.0
    yaw_ade = 0.0
    yaw_fde = 0.0
    vx_ae = 0.0
    vy_ae = 0.0
    wz_ae = 0.0
    control_ade = 0.0
    batch_cnt = 0
    sample_cnt = 0

    mpc = KoopmanMPC(
        dkm=dkm,
        state_dim=6,
        control_dim=12,  # 保持 12 维控制，在代价中加入软约束
        horizon=mpc_horizon,
        device=DEVICE,
    )

    first_batch_viz_done = False

    for batch_idx, batch in enumerate(data_loader):

        with torch.no_grad():
            # 真实状态（仅用于构造 Koopman 预测参考与误差统计）
            states = torch.stack(
                [
                    batch["x_est_gt"],
                    batch["y_est_gt"],
                    batch["thetaz_est_gt"],
                    batch["vx_est_gt"],
                    batch["vy_est_gt"],
                    batch["wz_est_gt"],
                ],
                dim=-1,
            ).to(DEVICE)  # [B, T+1, 6]

            # 控制输入（用于 Koopman 开环递推 + MPC warm start）
            controls = torch.cat(
                [batch["T_motor"]["total"], batch["delta_act"]["total"]], dim=-1
            ).to(DEVICE)  # [B, T, 12]
            controls_norm = (controls - CONTROL_MIN) / (CONTROL_MAX - CONTROL_MIN + 1e-6)

            # -------- 使用 Koopman 算子根据 states 与 controls 做开环递推，生成参考轨迹 -------- #
            # 与 eval_sequence 中保持一致：先将位置平移到序列起点坐标系
            states_rel = states.clone()
            states_rel[:, :, :2] -= states[:, 0:1, :2]

            # 编码并在 Koopman 空间中前向递推
            state_embeds_res = encoder(states_rel)  # [B, T+1, hidden_dim-6]
            state_embeds_full = torch.cat([states_rel, state_embeds_res], dim=-1)  # [B, T+1, hidden_dim]

            B, T_plus_1, hidden_dim = state_embeds_full.shape
            T = T_plus_1 - 1
            ITERS = T
            koopman_rollout = dkm(
                state_embeds_full[:, :-1, :], controls_norm, iters=ITERS
            )  # [B, T, hidden_dim]
            koop_states_rel = koopman_rollout[..., :6]  # [B, T, 6]，相对坐标系

            # 合并初始真实状态（相对坐标为 0）得到完整参考轨迹（T+1 个时刻）
            koop_states_full_rel = torch.cat(
                [states_rel[:, 0:1, :], koop_states_rel], dim=1
            )  # [B, T+1, 6]

            # 转回全局坐标，用于 MPC 跟踪参考
            koop_states_full = koop_states_full_rel.clone()
            koop_states_full[:, :, :2] += states[:, 0:1, :2]

        B, T_plus_1, _ = koop_states_full.shape
        assert B == 1, "当前闭环评估实现假设 batch_size = 1"

        # 初始状态（参考轨迹起点）
        s_t = koop_states_full[:, 0, :].unsqueeze(1)  # [B, 1, 6]

        # 在整段轨迹上做闭环 MPC
        cl_states = [s_t]  # list of [B, 1, 6]
        applied_controls = []  # list of [B,1,12]
        z_t = None
        u_init = None
        s_curr = koop_states_full[:, 0, :].unsqueeze(1) # [B,1,6]

        for t in tqdm.tqdm(range(T_plus_1 - 1)):
            # 剩余步长与 MPC 预测域取最小
            H = min(mpc_horizon, T_plus_1 - 1 - t)

            # 以当前真实状态为原点的局部坐标系
            with torch.no_grad():
                ref_traj = koop_states_full[:, t + 1 : t + 1 + H, :]     # [B,H,6]
                ref_traj_norm = ref_traj.clone()
                ref_traj_norm[:, :, :2] -= s_curr[:, 0:1, :2]
                s_curr_norm = s_curr.clone()
                s_curr_norm[:, :, :2] -= s_curr[:, 0:1, :2]

                # Koopman 初始状态
                z_t = torch.cat([s_curr_norm, encoder(s_curr_norm)], dim=-1)  # [B,1,hidden]
                z_ref = torch.cat([ref_traj_norm, encoder(ref_traj_norm)], dim=-1)  # [B,H,hidden]
                
            # 传统 QP MPC 求解（在 12 维控制空间）
            if u_init is not None:
                u_init = u_init[:, -(ref_traj.shape[1]):, :]
            U_opt = mpc.optimize(z_t.squeeze(1), z_ref, u_init=u_init)  # [B,H,12]
            u_init = U_opt

            # 施加当前时刻控制，向前一步
            u_apply = U_opt[:, 0, :]  # [B,12]
            applied_controls.append(u_apply.unsqueeze(1))  # [B,1,12]
            with torch.no_grad():
                # 使用原始 Koopman 动力学：z_{k+1} = z_k A + u_k B
                z_next = z_t.squeeze(1) @ dkm.A + u_apply @ dkm.B  # [B,16]
                x_next = z_next[..., :6]                           # [B,6]
                x_next = x_next.unsqueeze(1)
                x_next[:, :, :2] += s_curr[:, 0:1, :2]
            s_curr = x_next

            cl_states.append(x_next)

        cl_states = torch.cat(cl_states, dim=1)  # [B, T+1, 6]
        applied_controls = torch.cat(applied_controls, dim=1)  # [B, T, 12]

        # 计算轨迹跟踪误差
        with torch.no_grad():
            # 位置、航向误差（MPC 轨迹相对于 Koopman 参考轨迹；从 t=1 开始，避免初始态）
            pos_err = torch.norm(
                cl_states[:, 1:, :2] - koop_states_full[:, 1:, :2], p=2, dim=-1
            )  # [B, T]
            yaw_err = torch.abs(
                cl_states[:, 1:, 2] - koop_states_full[:, 1:, 2]
            )  # [B, T]

            traj_ade += pos_err.sum()
            yaw_ade += yaw_err.sum()

            # 终点误差
            fde_pos = torch.norm(
                cl_states[:, -1, :2] - koop_states_full[:, -1, :2], p=2, dim=-1
            )  # [B]
            fde_yaw = torch.abs(
                cl_states[:, -1, 2] - koop_states_full[:, -1, 2]
            )  # [B]

            traj_fde += fde_pos.sum()
            yaw_fde += fde_yaw.sum()

            # 控制误差：12 维归一化控制下的 L2 误差
            u_gt_full = controls_norm  # [B,T,12]
            u_mpc_full = applied_controls  # [B,T,12]
            control_err = torch.norm(u_mpc_full - u_gt_full, p=2, dim=-1)  # [B,T]
            control_ade += control_err.sum()

            # 速度相关误差（同样以 Koopman 预测为参考）
            vx_ae += torch.abs(
                cl_states[:, 1:, 3] - koop_states_full[:, 1:, 3]
            ).sum()
            vy_ae += torch.abs(
                cl_states[:, 1:, 4] - koop_states_full[:, 1:, 4]
            ).sum()
            wz_ae += torch.abs(
                cl_states[:, 1:, 5] - koop_states_full[:, 1:, 5]
            ).sum()

            batch_cnt += states.shape[0]
            sample_cnt += states.shape[0] * (T_plus_1 - 1)

        # 仅对首个 batch 进行可视化
        if (not first_batch_viz_done) and B == 1:
            first_batch_viz_done = True

            cl_np = cl_states[0].detach().cpu().numpy()         # [T+1, 6]
            ref_np = koop_states_full[0].detach().cpu().numpy() # [T+1, 6]

            # 真值 12 维控制 & MPC 12 维控制
            u_gt_np = controls_norm[0].detach().cpu().numpy()         # [T,12]
            u_mpc_np = applied_controls[0].detach().cpu().numpy()     # [T,12]

            time = np.arange(T_plus_1) * 0.01  # 与 eval_sequence 中保持一致

            # 1) XY 轨迹对比
            plt.figure(figsize=(6, 5))
            plt.plot(ref_np[:, 0], ref_np[:, 1], "b-", label="reference (Koopman)")
            plt.plot(cl_np[:, 0], cl_np[:, 1], "r--o", markersize=2, label="MPC closed-loop")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.title("Trajectory tracking (XY)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("mpc_traj_xy.png", dpi=200)
            plt.close()

            # 2) 关键状态量随时间：x, y, yaw
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

            axes[0].plot(time, ref_np[:, 0], "b-", label="ref x")
            axes[0].plot(time, cl_np[:, 0], "r--", label="mpc x")
            axes[0].set_ylabel("x [m]")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(time, ref_np[:, 1], "b-", label="ref y")
            axes[1].plot(time, cl_np[:, 1], "r--", label="mpc y")
            axes[1].set_ylabel("y [m]")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            axes[2].plot(time, ref_np[:, 2], "b-", label="ref yaw")
            axes[2].plot(time, cl_np[:, 2], "r--", label="mpc yaw")
            axes[2].set_xlabel("time [s]")
            axes[2].set_ylabel("yaw [rad]")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()
            plt.savefig("mpc_states_time.png", dpi=200)
            plt.close()

            # 3) 控制信号：12 维（6 个扭矩、6 个转角），同时画出真值控制与 MPC 控制
            ctrl_time = np.arange(u_mpc_np.shape[0]) * 0.01
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            u_gt_np = u_gt_np * (CONTROL_MAX.cpu().numpy() - CONTROL_MIN.cpu().numpy() + 1e-6) + CONTROL_MIN.cpu().numpy()
            u_mpc_np = u_mpc_np * (CONTROL_MAX.cpu().numpy() - CONTROL_MIN.cpu().numpy() + 1e-6) + CONTROL_MIN.cpu().numpy()

            # 扭矩：实线为真值，虚线为 MPC
            axes[0].plot(ctrl_time, u_gt_np[:, 0], "b-",  label="T_FL gt")
            axes[0].plot(ctrl_time, u_gt_np[:, 1], "g-",  label="T_FR gt")
            axes[0].plot(ctrl_time, u_gt_np[:, 2], "r-",  label="T_ML gt")
            axes[0].plot(ctrl_time, u_gt_np[:, 3], "c-",  label="T_MR gt")
            axes[0].plot(ctrl_time, u_gt_np[:, 4], "m-",  label="T_RL gt")
            axes[0].plot(ctrl_time, u_gt_np[:, 5], "y-",  label="T_RR gt")

            axes[0].plot(ctrl_time, u_mpc_np[:, 0], "b--",  label="T_FL mpc")
            axes[0].plot(ctrl_time, u_mpc_np[:, 1], "g--",  label="T_FR mpc")
            axes[0].plot(ctrl_time, u_mpc_np[:, 2], "r--",  label="T_ML mpc")
            axes[0].plot(ctrl_time, u_mpc_np[:, 3], "c--",  label="T_MR mpc")
            axes[0].plot(ctrl_time, u_mpc_np[:, 4], "m--",  label="T_RL mpc")
            axes[0].plot(ctrl_time, u_mpc_np[:, 5], "y--",  label="T_RR mpc")

            axes[0].set_ylabel("normalized torque")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(ncol=3, fontsize=7)

            # 转角：实线为真值，虚线为 MPC
            axes[1].plot(ctrl_time, u_gt_np[:, 6], "b-", label="delta_FL gt")
            axes[1].plot(ctrl_time, u_gt_np[:, 7], "g-", label="delta_FR gt")
            axes[1].plot(ctrl_time, u_gt_np[:, 8], "r-", label="delta_ML gt")
            axes[1].plot(ctrl_time, u_gt_np[:, 9], "c-", label="delta_MR gt")
            axes[1].plot(ctrl_time, u_gt_np[:, 10], "m-", label="delta_RL gt")
            axes[1].plot(ctrl_time, u_gt_np[:, 11], "y-", label="delta_RR gt")

            axes[1].plot(ctrl_time, u_mpc_np[:, 6], "b--", label="delta_FL mpc")
            axes[1].plot(ctrl_time, u_mpc_np[:, 7], "g--", label="delta_FR mpc")
            axes[1].plot(ctrl_time, u_mpc_np[:, 8], "r--", label="delta_ML mpc")
            axes[1].plot(ctrl_time, u_mpc_np[:, 9], "c--", label="delta_MR mpc")
            axes[1].plot(ctrl_time, u_mpc_np[:, 10], "m--", label="delta_RL mpc")
            axes[1].plot(ctrl_time, u_mpc_np[:, 11], "y--", label="delta_RR mpc")
            axes[1].set_xlabel("time [s]")
            axes[1].set_ylabel("normalized steer")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(ncol=3, fontsize=7)

            plt.tight_layout()
            plt.savefig("mpc_controls_time.png", dpi=200)
            plt.close()

        break

    # 归一化
    traj_ade /= sample_cnt
    traj_fde /= batch_cnt
    yaw_ade /= sample_cnt
    yaw_fde /= batch_cnt
    vx_ae /= sample_cnt
    vy_ae /= sample_cnt
    wz_ae /= sample_cnt
    control_ade /= sample_cnt

    # 将航向与横摆角速度误差转换为角度制输出
    yaw_ade_deg = yaw_ade * (180.0 / np.pi)
    yaw_fde_deg = yaw_fde * (180.0 / np.pi)
    wz_ae_deg = wz_ae * (180.0 / np.pi)

    print("MPC closed-loop tracking on test set:")
    print("Trajectory ADE: %.4f m, FDE: %.4f m" % (traj_ade.item(), traj_fde.item()))
    print(
        "Yaw ADE: %.4f deg, FDE: %.4f deg"
        % (yaw_ade_deg.item(), yaw_fde_deg.item())
    )
    print(
        "vx AE: %.4f m/s, vy AE: %.4f m/s, wz AE: %.4f deg/s"
        % (vx_ae.item(), vy_ae.item(), wz_ae_deg.item())
    )
    print("Control ADE: %.4f" % (control_ade.item()))

    return {
        "traj_ade": traj_ade.item(),
        "traj_fde": traj_fde.item(),
        "yaw_ade_deg": yaw_ade_deg.item(),
        "yaw_fde_deg": yaw_fde_deg.item(),
        "vx_ae": vx_ae.item(),
        "vy_ae": vy_ae.item(),
        "wz_ae_deg": wz_ae_deg.item(),
        "control_ade": control_ade.item(),
    }   


if __name__ == "__main__":
    # ----------------- 配置区（可按需修改） ----------------- #
    CKPT_PATH = "mpc/DeepEDMD-Transv2wonorm-hd16-multiset-100e-remote-local-lr1e-4-rollover-0.05pilossv24-0222.pth"
    EVAL_CONDITION = "all_wheel_steer"  # 与 eval_sequence.py 默认一致
    EVAL_LENGTH = 300                # 单条序列长度（控制步数）-1表示所有序列
    BATCH_SIZE = 1                  # MPC 计算较重，batch 不宜太大

    MPC_HORIZON = 30        # MPC 预测域长度
    # ---------------------------------------------------- #

    # 1) 加载模型
    encoder, dkm = build_model(CKPT_PATH)

    # 2) 构造测试数据加载器
    valid_loader = prepare_valid_loader(
        eval_condition=EVAL_CONDITION,
        eval_length=EVAL_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # 3) 基于 Koopman 模型进行 MPC 闭环轨迹跟踪评估
    mpc_closed_loop_eval(
        encoder=encoder,
        dkm=dkm,
        data_loader=valid_loader,
        mpc_horizon=MPC_HORIZON,
    )

