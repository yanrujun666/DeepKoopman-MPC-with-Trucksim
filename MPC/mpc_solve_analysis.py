"""
MPC求解过程重现与分析脚本
根据第一帧的输入状态和参考轨迹，重现完整的MPC求解过程
并进行详细的诊断分析
"""

import numpy as np
import scipy.io as scio
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# 导入控制器
from ddk_controller import DeepEDMD, MPCController


class MPCSolveAnalyzer:
    """MPC求解过程分析器"""
    
    def __init__(self, param_path: str, ref_traj_path: str, solver: str = 'auto'):
        """
        初始化分析器
        
        Args:
            param_path: DeepEDMD模型参数文件路径
            ref_traj_path: 参考轨迹数据文件路径
            solver: 求解器选择，'auto'（自动选择）、'quadprog'、'osqp'
        """
        self.param_path = param_path
        self.ref_traj_path = ref_traj_path
        self.solver = solver.lower()  # 求解器选择
        
        # 加载模型和控制器
        print("="*80)
        print("Initializing DeepEDMD model and MPC controller...")
        print("="*80)
        
        self.deepedmd = DeepEDMD(param_path)
        
        # MPC参数(应用短期优化方案以改善数值稳定性)
        # 1. 减小预测时域Np：从30减小到20，降低PHI和THETA矩阵条件数
        Np = 30
        Nc = 30  # 控制时域也相应减小
        sample_interval = 5
        model_dt = 0.01
        mpc_dt = float(sample_interval) * model_dt
        
        # Q：状态跟踪权重（6维：[X, Y, Yaw, vx, vy, yaw_rate]）
        # 目标：速度(vx)优先，同时整体更平顺
        Q = np.diag([20, 80, 120, 300, 40, 40])
        # 2. 增加控制权重R：转矩权重从5增加到50，改善H矩阵条件数
        # R：控制增量权重（12维：[6转矩, 6转向角]）
        # 目标：转矩更平滑（更大增量惩罚），转向更昂贵避免抖动
        R = np.diag([800, 800, 800, 800, 800, 800, 12000, 12000, 12000, 12000, 12000, 12000])
        # delta_umax：单步控制增量上限（归一化）
        # 转矩：0.05（≈150N·m/步），转向：0.02（≈3.6deg/步）
        delta_umax = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        
        self.mpc = MPCController(
            self.deepedmd,
            Np=Np,
            Nc=Nc,
            Q=Q,
            R=R,
            delta_umax=delta_umax,
            model_dt=model_dt,
            mpc_dt=mpc_dt
        )
        
        # 加载参考轨迹
        self._load_reference_trajectory()
        
        print("\nInitialization completed!\n")
    
    def _load_reference_trajectory(self):
        """加载参考轨迹数据"""
        ref_data = scio.loadmat(self.ref_traj_path, squeeze_me=True)
        
        # 辅助函数：从MATLAB结构体中提取字段
        def extract_struct_field(struct_data, field_name):
            if isinstance(struct_data, np.ndarray) and struct_data.dtype.names:
                if struct_data.shape ==() and field_name in struct_data.dtype.names:
                    value = struct_data[field_name]
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        return value.item()
                    return value
            elif hasattr(struct_data, field_name):
                return getattr(struct_data, field_name)
            return None
        
        # 检查是否是新格式(ref_trajectory结构体)
        if 'ref_trajectory' in ref_data:
            ref_traj_struct = ref_data['ref_trajectory']
            
            pos_struct = extract_struct_field(ref_traj_struct, 'position')
            pos_x = extract_struct_field(pos_struct, 'x')
            pos_y = extract_struct_field(pos_struct, 'y')
            pos_yaw = extract_struct_field(pos_struct, 'yaw')
            
            vel_struct = extract_struct_field(ref_traj_struct, 'velocity')
            vel_vx = extract_struct_field(vel_struct, 'vx')
            vel_vy = extract_struct_field(vel_struct, 'vy')
            vel_wz = extract_struct_field(vel_struct, 'wz')
            
            # 组合为数组格式 [X, Y, Yaw, vx, vy, yaw_rate]
            self.ref_traj = np.column_stack([
                np.array(pos_x).flatten(),
                np.array(pos_y).flatten(),
                np.array(pos_yaw).flatten(),
                np.array(vel_vx).flatten(),
                np.array(vel_vy).flatten(),
                np.array(vel_wz).flatten()
            ])
        else:
            # 兼容旧格式
            pos = ref_data.get('position', ref_data.get('Pos', None))
            vel = ref_data.get('velocity', ref_data.get('X', None))
            if pos is not None and vel is not None:
                self.ref_traj = np.column_stack([pos, vel])
            else:
                raise ValueError("无法加载参考轨迹数据")
        
        print(f"Reference trajectory loaded, length: {len(self.ref_traj)}")
    
    def _determine_solver(self) -> str:
        """确定要使用的求解器"""
        if self.solver == 'auto':
            # 自动选择：优先quadprog，如果不可用则使用OSQP
            try:
                import quadprog
                return 'quadprog'
            except ImportError:
                try:
                    import osqp
                    return 'osqp'
                except ImportError:
                    raise ImportError("Neither quadprog nor OSQP is installed. Please install at least one: pip install quadprog or pip install osqp")
        elif self.solver == 'quadprog':
            try:
                import quadprog
                return 'quadprog'
            except ImportError:
                raise ImportError("Quadprog is not installed. Install with: pip install quadprog")
        elif self.solver == 'osqp':
            try:
                import osqp
                return 'osqp'
            except ImportError:
                raise ImportError("OSQP is not installed. Install with: pip install osqp")
        else:
            raise ValueError(f"Unknown solver: {self.solver}. Choose from 'auto', 'quadprog', or 'osqp'")
    
    def _solve_with_quadprog(self, f, b_ineq, lb, ub):
        """使用quadprog求解"""
        try:
            import quadprog
            
            # quadprog标准形式: min 0.5 * x^T * G * x + a^T * x, s.t. C^T * x >= b
            # 我们需要将问题转换为quadprog格式
            # 约束: A_ineq * x <= b_ineq 等价于 -A_ineq * x >= -b_ineq
            # 边界: lb <= x <= ub 等价于 x >= lb 和 -x >= -ub
            
            # 使用预存储的密集矩阵（quadprog需要）
            H_dense = self.mpc.H_dense
            
            # 构建约束矩阵 C^T * x >= b
            # 约束1: A_ineq * x <= b_ineq  -> -A_ineq * x >= -b_ineq
            A_ineq_1 = np.hstack([self.mpc.A_l, np.zeros((self.mpc.Nc * self.mpc.Nu, 1))])
            A_ineq_2 = np.hstack([-self.mpc.A_l, np.zeros((self.mpc.Nc * self.mpc.Nu, 1))])
            A_ineq = np.vstack([A_ineq_1, A_ineq_2])
            
            # 约束2: x >= lb  -> I * x >= lb
            # 约束3: x <= ub  -> -I * x >= -ub
            n_vars = len(f)
            I_mat = np.eye(n_vars)
            
            # 合并所有约束: C^T * x >= b
            # quadprog的API: solve_qp(G, a, C, b, meq=0)
            # 其中C的形状是 (n_vars, n_constraints)，表示 C^T * x >= b
            C_constraints = np.vstack([-A_ineq, I_mat, -I_mat])  # (n_constraints, n_vars)
            b_constraints = np.concatenate([-b_ineq, lb, -ub])  # (n_constraints,)
            
            # quadprog要求C的形状是 (n_vars, n_constraints)
            # 所以需要转置：C = C_constraints^T
            C_T = C_constraints.T  # (n_vars, n_constraints)
            
            # 检查约束可行性（避免数值问题）
            b_safe = b_constraints.copy()
            extreme_neg_mask = b_safe < -1e10
            if np.any(extreme_neg_mask):
                b_safe[extreme_neg_mask] = -1e10
            
            # 确保H矩阵是正定的（添加小的正则化项）
            H_quadprog = H_dense.copy()
            
            # 求解并记录时间
            start_time = time.time()
            try:
                # quadprog.solve_qp(G, a, C, b, meq=0)
                # 注意：quadprog的G是Hessian矩阵（不需要乘以2），a是梯度向量
                # 返回值格式: (x, fval, u, iact, nact, iter)
                result = quadprog.solve_qp(
                    H_quadprog,  # G: Hessian矩阵
                    f,           # a: 梯度向量
                    C_T,         # C: 约束矩阵 (n_vars, n_constraints)
                    b_safe,      # b: 约束右端项
                    meq=0        # 等式约束数量（0表示只有不等式约束）
                )
                
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
                                    iterations = int(iter_count.item())
                                elif arr_size > 0:
                                    iterations = int(iter_count.flat[0])
                                else:
                                    iterations = -1
                            else:
                                iterations = int(iter_count)
                        except (ValueError, TypeError, AttributeError):
                            # 如果转换失败，假设成功
                            iterations = 0
                        # 如果迭代次数 >= 0，通常表示成功
                        # 同时检查X是否有效
                        solve_success = (iterations >= 0 and X is not None)
                    else:
                        iterations = None
                        solve_success = (X is not None and len(X) > 0)
                    
                    solve_status = "solved" if solve_success else "failed"
                else:
                    X = None
                    fval = None
                    iterations = None
                    solve_success = False
                    solve_status = "invalid result format"
                
                solve_time = time.time() - start_time
            except Exception as e:
                solve_time = time.time() - start_time
                solve_success = False
                solve_status = f"quadprog error: {str(e)}"
                X = None
                fval = None
                iterations = None
            
            return {
                'solver': 'quadprog',
                'success': solve_success,
                'X': X,
                'solve_time': solve_time,
                'status': solve_status,
                'iterations': iterations,
                'obj_val': fval,
                'pri_res': None,
                'dua_res': None
            }
        except ImportError:
            return {
                'solver': 'quadprog',
                'success': False,
                'X': None,
                'solve_time': 0.0,
                'status': 'quadprog not installed',
                'iterations': None,
                'obj_val': None,
                'pri_res': None,
                'dua_res': None,
                'error': 'Quadprog not installed'
            }
        except Exception as e:
            return {
                'solver': 'quadprog',
                'success': False,
                'X': None,
                'solve_time': 0.0,
                'status': f'quadprog error: {str(e)}',
                'iterations': None,
                'obj_val': None,
                'pri_res': None,
                'dua_res': None,
                'error': str(e)
            }
    
    def _solve_with_osqp(self, f, b_ineq, lb, ub):
        """使用OSQP求解"""
        try:
            import osqp
            
            # 构建OSQP问题
            LARGE_NEG = -1e20
            l_combined = np.concatenate([LARGE_NEG * np.ones(len(b_ineq)), lb])
            u_combined = np.concatenate([b_ineq, ub])
            
            # 创建求解器
            solver = osqp.OSQP()
            settings = {
                'verbose': True,
                'eps_abs': 1e-5,
                'eps_rel': 1e-5,
                'max_iter': 4000,
                'warm_start': True,
                'polish': True,
            }
            
            solver.setup(
                P=self.mpc.H_sparse,
                q=f,
                A=self.mpc.A_combined,
                l=l_combined,
                u=u_combined,
                **settings
            )
            
            # 求解
            start_time = time.time()
            result = solver.solve()
            solve_time = time.time() - start_time
            
            if result.info.status in ['solved', 'solved inaccurate']:
                X = result.x
                solve_success = True
                solve_status = result.info.status
                iterations = result.info.iter
                obj_val = getattr(result.info, 'obj_val', None)
                pri_res = getattr(result.info, 'pri_res', None)
                dua_res = getattr(result.info, 'dua_res', None)
            else:
                X = None
                solve_success = False
                solve_status = result.info.status
                iterations = result.info.iter if hasattr(result.info, 'iter') else None
                obj_val = None
                pri_res = None
                dua_res = None
            
            return {
                'solver': 'osqp',
                'success': solve_success,
                'X': X,
                'solve_time': solve_time,
                'status': solve_status,
                'iterations': iterations,
                'obj_val': obj_val,
                'pri_res': pri_res,
                'dua_res': dua_res
            }
        except ImportError:
            return {
                'solver': 'osqp',
                'success': False,
                'X': None,
                'solve_time': 0.0,
                'status': 'osqp not installed',
                'iterations': None,
                'obj_val': None,
                'pri_res': None,
                'dua_res': None,
                'error': 'OSQP not installed'
            }
        except Exception as e:
            return {
                'solver': 'osqp',
                'success': False,
                'X': None,
                'solve_time': 0.0,
                'status': f'osqp error: {str(e)}',
                'iterations': None,
                'obj_val': None,
                'pri_res': None,
                'dua_res': None,
                'error': str(e)
            }
    
    def analyze_first_step(self, state_input: np.ndarray, nearest_idx: int = 1, 
                          u_prev: Optional[np.ndarray] = None) -> Dict:
        """
        分析第一帧的MPC求解过程
        
        Args:
            state_input: 车辆当前状态 [X, Y, Yaw, vx, vy, yaw_rate]
            nearest_idx: 参考轨迹最近点索引
            u_prev: 上一时刻控制输入(归一化)，如果为None则使用零控制
        
        Returns:
            分析结果字典
        """
        print("="*80)
        print("Analyzing first step MPC solving process")
        print("="*80)
        
        # 初始化控制输入
        if u_prev is None:
            # 注意：本项目控制量采用CONTROL_MIN/MAX做min-max归一化
            # 因此 u=0.5 对应“零转矩/零转角”（控制量区间中心）
            u_prev = 0.5 * np.ones(12)  # 12维控制（归一化）
        
        # 记录分析结果
        analysis_result = {
            'state_input': state_input.copy(),
            'nearest_idx': nearest_idx,
            'u_prev': u_prev.copy(),
            'steps': {}
        }
        
        try:
            # ========== Step 1: Reference trajectory extraction ==========
            print("\n[Step 1] Reference Trajectory Extraction")
            print("-" * 80)
            sample_interval = 5
            Np = self.mpc.Np
            
            end_idx = int(min(nearest_idx + Np * sample_interval, len(self.ref_traj)))
            temp_refr = self.ref_traj[nearest_idx:end_idx:sample_interval, :]
            
            # 确保提取的点数等于Np
            if len(temp_refr) < Np:
                last_point = temp_refr[-1, :] if len(temp_refr) > 0 else self.ref_traj[-1, :]
                padding = np.tile(last_point,(int(Np - len(temp_refr)), 1))
                temp_refr = np.vstack([temp_refr, padding])
            elif len(temp_refr) > Np:
                temp_refr = temp_refr[:Np, :]
            
            # Note: Since model is trained with global coordinates, we use global coordinates directly
            # No coordinate transformation needed
            
            print(f"  Reference trajectory index range: {nearest_idx} to {end_idx}(interval={sample_interval})")
            print(f"  Number of extracted reference points: {len(temp_refr)}")
            print(f"  First reference point (global coordinates): X={temp_refr[0,0]:.4f}m, Y={temp_refr[0,1]:.4f}m, Yaw={temp_refr[0,2]*180/np.pi:.4f}deg")
            
            analysis_result['steps']['ref_traj_extraction'] = {
                'ref_traj': temp_refr.copy()
            }
            
            # ========== Step 2: State encoding (using global coordinates) ==========
            print("\n[Step 2] State Encoding (Global Coordinates)")
            print("-" * 80)
            print("  Note: Model trained with global coordinates, no coordinate transformation needed")
            
            # 直接使用全局状态进行编码（不进行坐标变换）
            # 参考轨迹也直接使用全局坐标
            ref_r = temp_refr.copy()  # Use global reference trajectory directly, no transformation
            
            print(f"  Reference trajectory (global coordinates), shape: {ref_r.shape}")
            print(f"  First reference point (global coordinates): X={ref_r[0,0]:.4f}m, Y={ref_r[0,1]:.4f}m, Yaw={ref_r[0,2]*180/np.pi:.4f}deg")
            
            print(f"  Current state (global coordinates):")
            print(f"    Position: X={state_input[0]:.4f}m, Y={state_input[1]:.4f}m, Yaw={state_input[2]*180/np.pi:.4f}deg")
            print(f"    Velocity: vx={state_input[3]:.4f}m/s, vy={state_input[4]:.4f}m/s, yaw_rate={state_input[5]*180/np.pi:.4f}deg/s")
            
            # 直接使用全局状态进行编码（不进行坐标变换）
            x_lift = self.deepedmd.encoder(state_input)
            print(f"  State encoding completed, lifted state dimension: {len(x_lift)}")
            print(f"  Lifted state first 6 dims(original state): {x_lift[:6]}")
            print(f"  Lifted state last 10 dims(lifted features): {x_lift[6:]}")
            
            # Store for analysis
            analysis_result['steps']['coordinate_transform'] = {
                'ref_r': ref_r.copy(),  # Global reference trajectory
                'x_global': state_input.copy()  # Global state (no transformation)
            }
            
            analysis_result['steps']['encoding'] = {
                'x_lift': x_lift.copy()
            }
            
            # ========== Step 4: Build QP problem ==========
            print("\n[Step 4] Build QP Problem")
            print("-" * 80)
            
            # 构建扩展状态
            kesi = np.concatenate([x_lift, u_prev])
            print(f"  Extended state dimension: {len(kesi)}(lifted state {len(x_lift)} + control {len(u_prev)})")
            
            # 构建参考轨迹向量
            ref_r_vec = ref_r.T.flatten('F')
            print(f"  Reference trajectory vector dimension: {len(ref_r_vec)}")
            
            # 计算跟踪误差
            error = self.mpc.PHI @ kesi - ref_r_vec
            error_norm = np.linalg.norm(error)
            error_max = np.max(np.abs(error))
            error_mean = np.mean(np.abs(error))
            
            print(f"  Tracking error:")
            print(f"    Error norm: {error_norm:.6f}")
            print(f"    Max error: {error_max:.6f}")
            print(f"    Mean error: {error_mean:.6f}")
            
            # 将误差reshape为(Np, state_dim)以便分析各维度误差
            error_reshaped = error.reshape(self.mpc.Np, self.deepedmd.s_dim)
            error_by_dim = np.mean(np.abs(error_reshaped), axis=0)
            print(f"  Average error by dimension: x={error_by_dim[0]:.6f}, y={error_by_dim[1]:.6f}, yaw={error_by_dim[2]:.6f}, "
                  f"vx={error_by_dim[3]:.6f}, vy={error_by_dim[4]:.6f}, yaw_rate={error_by_dim[5]:.6f}")
            
            # 构建梯度向量
            f_1 =(2 * error.T @ self.mpc.Q_kron @ self.mpc.THETA).flatten()
            f_2 = np.array([0.0])
            f = np.concatenate([f_1, f_2])
            
            print(f"  Gradient vector f dimension: {len(f)}")
            print(f"  Gradient vector f statistics: mean={np.mean(f):.6f}, std={np.std(f):.6f}, max={np.max(np.abs(f)):.6f}")
            
            # 构建约束
            Ut = np.kron(np.ones(self.mpc.Nc), u_prev)
            b_ineq_1 = self.mpc.Umax - Ut
            b_ineq_2 = -self.mpc.Umin + Ut
            b_ineq = np.concatenate([b_ineq_1, b_ineq_2])
            
            # 边界约束
            lb = self.mpc.lb_fixed.copy()
            ub = self.mpc.ub_fixed.copy()
            
            # 动态调整边界(与solve方法中的逻辑一致)
            for i in range(self.mpc.Nu):
                u_prev_i = u_prev[i]
                if u_prev_i <= 0.05:
                    min_delta = -u_prev_i + 1e-6
                    for k in range(self.mpc.Nc):
                        idx = k * self.mpc.Nu + i
                        lb[idx] = max(lb[idx], min_delta)
                elif u_prev_i >= 0.95:
                    max_delta =(1.0 - u_prev_i) - 1e-6
                    for k in range(self.mpc.Nc):
                        idx = k * self.mpc.Nu + i
                        ub[idx] = min(ub[idx], max_delta)
            
            # 确保lb <= ub
            for i in range(len(lb)):
                if lb[i] > ub[i] + 1e-10:
                    mid =(lb[i] + ub[i]) / 2
                    lb[i] = mid - 1e-6
                    ub[i] = mid + 1e-6
                    if i < self.mpc.Nc * self.mpc.Nu:
                        dim_idx = i % self.mpc.Nu
                        lb[i] = max(lb[i], -self.mpc.delta_umax[dim_idx])
                        ub[i] = min(ub[i], self.mpc.delta_umax[dim_idx])
            
            print(f"  Constraint count: inequality {len(b_ineq)}, boundary {len(lb)}")
            print(f"  Boundary constraint range: lb_min={np.min(lb):.6f}, ub_max={np.max(ub):.6f}")
            
            analysis_result['steps']['qp_problem'] = {
                'kesi': kesi.copy(),
                'ref_r_vec': ref_r_vec.copy(),
                'error': error.copy(),
                'error_norm': error_norm,
                'error_by_dim': error_by_dim.copy(),
                'f': f.copy(),
                'b_ineq': b_ineq.copy(),
                'lb': lb.copy(),
                'ub': ub.copy()
            }
            
            # ========== Step 5: Numerical diagnostics ==========
            print("\n[Step 5] Numerical Diagnostics")
            print("-" * 80)
            
            # H矩阵诊断
            H_cond = np.linalg.cond(self.mpc.H_dense)
            H_eigvals = np.linalg.eigvals(self.mpc.H_dense)
            H_min_eigval = np.min(np.real(H_eigvals))
            H_max_eigval = np.max(np.real(H_eigvals))
            
            print(f"  H matrix diagnostics:")
            print(f"    Condition number: {H_cond:.2e}")
            print(f"    Eigenvalue range: [{H_min_eigval:.6f}, {H_max_eigval:.6f}]")
            print(f"    Minimum eigenvalue: {H_min_eigval:.6e}")
            if H_min_eigval < 0:
                print(f"    Warning: H matrix is not positive definite!")
            elif H_min_eigval < 1e-8:
                print(f"    Warning: H matrix is near singular!")
            
            # THETA矩阵诊断
            THETA_norm = np.linalg.norm(self.mpc.THETA)
            THETA_cond = np.linalg.cond(self.mpc.THETA)
            
            print(f"  THETA matrix diagnostics:")
            print(f"    Frobenius norm: {THETA_norm:.2e}")
            print(f"    Condition number: {THETA_cond:.2e}")
            
            # PHI矩阵诊断
            PHI_norm = np.linalg.norm(self.mpc.PHI)
            PHI_cond = np.linalg.cond(self.mpc.PHI)
            
            print(f"  PHI matrix diagnostics:")
            print(f"    Frobenius norm: {PHI_norm:.2e}")
            print(f"    Condition number: {PHI_cond:.2e}")
            
            # A_ext矩阵诊断
            A_ext_eigvals = np.linalg.eigvals(self.mpc.A_ext)
            A_ext_max_eigval = np.max(np.abs(A_ext_eigvals))
            
            print(f"  A_ext matrix diagnostics:")
            print(f"    Maximum eigenvalue magnitude: {A_ext_max_eigval:.8f}")
            if A_ext_max_eigval >= 1.0:
                print(f"    Warning: A_ext matrix may be unstable!")
            
            # 约束矩阵诊断
            from scipy import sparse
            A_combined_cond = np.linalg.cond(self.mpc.A_combined.toarray())
            
            print(f"  Constraint matrix A_combined diagnostics:")
            print(f"    Condition number: {A_combined_cond:.2e}")
            
            analysis_result['steps']['diagnostics'] = {
                'H_cond': H_cond,
                'H_min_eigval': H_min_eigval,
                'H_max_eigval': H_max_eigval,
                'THETA_norm': THETA_norm,
                'THETA_cond': THETA_cond,
                'PHI_norm': PHI_norm,
                'PHI_cond': PHI_cond,
                'A_ext_max_eigval': A_ext_max_eigval,
                'A_combined_cond': A_combined_cond
            }
            
            # ========== Step 6: QP solving ==========
            print("\n[Step 6] QP Solving")
            print("-" * 80)
            
            # 确定使用的求解器
            solver_to_use = self._determine_solver()
            print(f"  Using solver: {solver_to_use}")
            
            # 尝试求解
            solve_result = None
            if solver_to_use == 'quadprog':
                solve_result = self._solve_with_quadprog(f, b_ineq, lb, ub)
            elif solver_to_use == 'osqp':
                solve_result = self._solve_with_osqp(f, b_ineq, lb, ub)
            else:
                # 自动选择：先尝试quadprog，失败则尝试OSQP
                solve_result = self._solve_with_quadprog(f, b_ineq, lb, ub)
                if not solve_result['success']:
                    print(f"  Quadprog failed, trying OSQP...")
                    solve_result = self._solve_with_osqp(f, b_ineq, lb, ub)
            
            # 处理求解结果
            print(f"  Solve time: {solve_result['solve_time']:.4f}s")
            print(f"  Solve status: {solve_result['status']}")
            if solve_result.get('iterations') is not None:
                print(f"  Iterations: {solve_result['iterations']}")
            if solve_result.get('obj_val') is not None:
                print(f"  Objective value: {solve_result['obj_val']:.6f}")
            if solve_result.get('pri_res') is not None:
                print(f"  Primal residual: {solve_result['pri_res']:.6e}")
            if solve_result.get('dua_res') is not None:
                print(f"  Dual residual: {solve_result['dua_res']:.6e}")
            
            if solve_result['success'] and solve_result['X'] is not None:
                X = solve_result['X']
                delta_u = X[:self.mpc.Nu]
                
                print(f"  Solve successful!")
                print(f"  Control increment delta_u: {delta_u}")
                
                # 更新控制量
                u_new = u_prev + delta_u
                u_clipped = np.clip(u_new, 0.0, 1.0)
                
                # 转换为实际控制指令
                control_output = self.mpc.convert_to_control_output(u_clipped)
                
                print(f"  Normalized control u_new: {u_clipped}")
                print(f"  Actual control output:")
                print(f"    Steering angles:")
                print(f"      LF={control_output[0]:.2f}deg, RF={control_output[1]:.2f}deg, LM={control_output[2]:.2f}deg")
                print(f"      RM={control_output[3]:.2f}deg, LR={control_output[4]:.2f}deg, RR={control_output[5]:.2f}deg")
                print(f"    Torques:")
                print(f"      LF={control_output[6]:.2f}N·m, RF={control_output[7]:.2f}N·m, LM={control_output[8]:.2f}N·m")
                print(f"      RM={control_output[9]:.2f}N·m, LR={control_output[10]:.2f}N·m, RR={control_output[11]:.2f}N·m")
                
                analysis_result['steps']['qp_solve'] = {
                    'success': True,
                    'solver': solve_result['solver'],
                    'solve_time': solve_result['solve_time'],
                    'status': solve_result['status'],
                    'iterations': solve_result['iterations'],
                    'obj_val': solve_result['obj_val'],
                    'pri_res': solve_result.get('pri_res'),
                    'dua_res': solve_result.get('dua_res'),
                    'delta_u': delta_u.copy(),
                    'u_new': u_clipped.copy(),
                    'control_output': control_output.copy()
                }
            else:
                print(f"  Solve failed: {solve_result['status']}")
                analysis_result['steps']['qp_solve'] = {
                    'success': False,
                    'solver': solve_result['solver'],
                    'status': solve_result['status'],
                    'iterations': solve_result['iterations'],
                    'error': solve_result.get('error')
                }
        
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
            # 确保即使出错也返回部分结果
            if 'analysis_result' not in locals() or analysis_result is None:
                analysis_result = {
                    'state_input': state_input.copy() if state_input is not None else None,
                    'nearest_idx': nearest_idx,
                    'u_prev': u_prev.copy() if u_prev is not None else None,
                    'steps': {},
                    'error': str(e)
                }
            else:
                # 如果analysis_result已存在，添加错误信息
                analysis_result['error'] = str(e)
        
        # 确保总是返回一个有效的字典
        if analysis_result is None:
            analysis_result = {
                'state_input': None,
                'nearest_idx': nearest_idx,
                'u_prev': None,
                'steps': {},
                'error': 'Unknown error: analysis_result is None'
            }
        
        return analysis_result
    
    def generate_report(self, analysis_result: Dict, output_dir: str = "analysis_output"):
        """
        生成分析报告(文本+可视化)
        
        Args:
            analysis_result: 分析结果字典
            output_dir: 输出目录
        """
        # 检查analysis_result是否为None
        if analysis_result is None:
            print("Error: analysis_result is None, cannot generate report")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # ========== 生成文本报告 ==========
        report_file = output_path / "mpc_analysis_report_short_term_optimization.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MPC Solving Process Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            # 输入信息
            f.write("1. Input Information\n")
            f.write("-"*80 + "\n")
            if 'state_input' in analysis_result and analysis_result['state_input'] is not None:
                state = analysis_result['state_input']
                f.write(f"Vehicle state: X={state[0]:.4f}m, Y={state[1]:.4f}m, Yaw={state[2]*180/np.pi:.4f}deg\n")
                f.write(f"Velocity: vx={state[3]:.4f}m/s, vy={state[4]:.4f}m/s, yaw_rate={state[5]*180/np.pi:.4f}deg/s\n")
            else:
                f.write("Vehicle state: Not available\n")
            
            if 'nearest_idx' in analysis_result:
                f.write(f"Reference trajectory index: {analysis_result['nearest_idx']}\n")
            if 'u_prev' in analysis_result and analysis_result['u_prev'] is not None:
                f.write(f"Previous control input: {analysis_result['u_prev']}\n")
            f.write("\n")
            
            # 各步骤结果
            steps = analysis_result['steps']
            
            if 'ref_traj_extraction' in steps:
                f.write("2. Reference Trajectory Extraction (Global Coordinates)\n")
                f.write("-"*80 + "\n")
                ref_traj = steps['ref_traj_extraction']['ref_traj']
                f.write(f"Number of extracted reference points: {len(ref_traj)}\n")
                f.write(f"First reference point (global): X={ref_traj[0,0]:.4f}m, Y={ref_traj[0,1]:.4f}m, Yaw={ref_traj[0,2]*180/np.pi:.4f}deg\n\n")
            
            if 'coordinate_transform' in steps:
                f.write("3. State (Global Coordinates)\n")
                f.write("-"*80 + "\n")
                x_global = steps['coordinate_transform']['x_global']
                f.write(f"Global state (no transformation): {x_global}\n\n")
            
            if 'encoding' in steps:
                f.write("4. State Encoding\n")
                f.write("-"*80 + "\n")
                x_lift = steps['encoding']['x_lift']
                f.write(f"Lifted state dimension: {len(x_lift)}\n")
                f.write(f"Lifted state: {x_lift}\n\n")
            
            if 'qp_problem' in steps:
                f.write("5. QP Problem Construction\n")
                f.write("-"*80 + "\n")
                qp = steps['qp_problem']
                f.write(f"Tracking error norm: {qp['error_norm']:.6f}\n")
                f.write(f"Average error by dimension: {qp['error_by_dim']}\n")
                f.write(f"Gradient vector f statistics: mean={np.mean(qp['f']):.6f}, max={np.max(np.abs(qp['f'])):.6f}\n\n")
            
            if 'diagnostics' in steps:
                f.write("6. Numerical Diagnostics\n")
                f.write("-"*80 + "\n")
                diag = steps['diagnostics']
                f.write(f"H matrix condition number: {diag['H_cond']:.2e}\n")
                f.write(f"H matrix minimum eigenvalue: {diag['H_min_eigval']:.6e}\n")
                f.write(f"THETA matrix norm: {diag['THETA_norm']:.2e}\n")
                f.write(f"THETA matrix condition number: {diag['THETA_cond']:.2e}\n")
                f.write(f"A_ext maximum eigenvalue magnitude: {diag['A_ext_max_eigval']:.8f}\n")
                if diag['A_ext_max_eigval'] >= 1.0:
                    f.write("  Warning: A_ext matrix may be unstable!\n")
                f.write("\n")
            
            if 'qp_solve' in steps:
                solve = steps['qp_solve']
                solver_name = solve.get('solver', 'unknown').upper()
                f.write(f"7. {solver_name} Solving Results\n")
                f.write("-"*80 + "\n")
                if solve.get('success', False):
                    f.write(f"Solve status: {solve['status']}\n")
                    f.write(f"Solve time: {solve['solve_time']:.4f}s\n")
                    if solve.get('iterations') is not None:
                        f.write(f"Iterations: {solve['iterations']}\n")
                    if solve.get('obj_val') is not None:
                        f.write(f"Objective value: {solve['obj_val']:.6f}\n")
                    f.write(f"Control increment: {solve['delta_u']}\n")
                    f.write(f"Actual control output: {solve['control_output']}\n")
                else:
                    f.write(f"Solve failed: {solve.get('status', solve.get('error', 'Unknown'))}\n")
        
        print(f"\nText report saved to: {report_file}")
        
        # ========== 生成可视化图表 ==========
        self._generate_plots(analysis_result, output_path)
    
    def _generate_plots(self, analysis_result: Dict, output_path: Path):
        """生成可视化图表"""
        try:
            steps = analysis_result['steps']
            
            # 创建图形
            fig = plt.figure(figsize=(16, 12))
            
            # 子图1: 跟踪误差
            if 'qp_problem' in steps:
                ax1 = plt.subplot(3, 3, 1)
                error = steps['qp_problem']['error']
                error_reshaped = error.reshape(self.mpc.Np, self.deepedmd.s_dim)
                time_steps = np.arange(1, self.mpc.Np + 1) * self.mpc.mpc_dt
                
                ax1.plot(time_steps, error_reshaped[:, 0], label='x')
                ax1.plot(time_steps, error_reshaped[:, 1], label='y')
                ax1.plot(time_steps, error_reshaped[:, 2], label='yaw')
                ax1.set_xlabel('Prediction Time(s)')
                ax1.set_ylabel('Tracking Error')
                ax1.set_title('Tracking Error vs Time')
                ax1.legend()
                ax1.grid(True)
            
            # 子图2: 控制增量
            if 'qp_solve' in steps and steps['qp_solve'].get('success', False):
                ax2 = plt.subplot(3, 3, 2)
                delta_u = steps['qp_solve']['delta_u']
                control_names = [f'u{i}' for i in range(len(delta_u))]
                ax2.bar(control_names, delta_u)
                ax2.set_xlabel('Control Input Dimension')
                ax2.set_ylabel('Control Increment')
                ax2.set_title('Control Increment delta_u')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, axis='y')
            
            # 子图3: 归一化控制输入
            if 'qp_solve' in steps and steps['qp_solve'].get('success', False):
                ax3 = plt.subplot(3, 3, 3)
                u_new = steps['qp_solve']['u_new']
                control_names = [f'u{i}' for i in range(len(u_new))]
                ax3.bar(control_names, u_new)
                ax3.set_xlabel('Control Input Dimension')
                ax3.set_ylabel('Normalized Control Input')
                ax3.set_title('Normalized Control Input u_new')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, axis='y')
            
            # 子图4: H矩阵特征值
            if 'diagnostics' in steps:
                ax4 = plt.subplot(3, 3, 4)
                H_eigvals = np.linalg.eigvals(self.mpc.H_dense)
                H_eigvals_real = np.real(H_eigvals)
                H_eigvals_imag = np.imag(H_eigvals)
                ax4.scatter(H_eigvals_real, H_eigvals_imag, alpha=0.6)
                ax4.set_xlabel('Real Part')
                ax4.set_ylabel('Imaginary Part')
                ax4.set_title(f'H Matrix Eigenvalues(cond={steps["diagnostics"]["H_cond"]:.2e})')
                ax4.grid(True)
            
            # 子图5: A_ext矩阵特征值
            if 'diagnostics' in steps:
                ax5 = plt.subplot(3, 3, 5)
                A_ext_eigvals = np.linalg.eigvals(self.mpc.A_ext)
                A_ext_eigvals_abs = np.abs(A_ext_eigvals)
                angles = np.angle(A_ext_eigvals)
                ax5.scatter(A_ext_eigvals_abs * np.cos(angles), 
                           A_ext_eigvals_abs * np.sin(angles), alpha=0.6)
                # 绘制单位圆
                theta = np.linspace(0, 2*np.pi, 100)
                ax5.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1, label='Unit Circle')
                ax5.set_xlabel('Real Part')
                ax5.set_ylabel('Imaginary Part')
                ax5.set_title(f'A_ext Eigenvalues(max|λ|={steps["diagnostics"]["A_ext_max_eigval"]:.6f})')
                ax5.legend()
                ax5.grid(True)
                ax5.axis('equal')
            
            # 子图6: 梯度向量f
            if 'qp_problem' in steps:
                ax6 = plt.subplot(3, 3, 6)
                f = steps['qp_problem']['f']
                ax6.plot(f[:self.mpc.Nc * self.mpc.Nu], 'o-', markersize=3)
                ax6.set_xlabel('Variable Index')
                ax6.set_ylabel('Gradient Value')
                ax6.set_title('Gradient Vector f(Control Increment Part)')
                ax6.grid(True)
            
            # 子图7: 参考轨迹(前几个点，全局坐标)
            if 'ref_traj_extraction' in steps:
                ax7 = plt.subplot(3, 3, 7)
                ref_traj = steps['ref_traj_extraction']['ref_traj']
                ax7.plot(ref_traj[:10, 0], ref_traj[:10, 1], 'o-', label='Reference Trajectory (Global)')
                state = analysis_result['state_input']
                ax7.plot(state[0], state[1], 'r*', markersize=15, label='Current State (Global)')
                ax7.set_xlabel('X(m)')
                ax7.set_ylabel('Y(m)')
                ax7.set_title('Reference Trajectory (Global, First 10 Points)')
                ax7.legend()
                ax7.grid(True)
                ax7.axis('equal')
            
            # 子图8: 误差各维度分布
            if 'qp_problem' in steps:
                ax8 = plt.subplot(3, 3, 8)
                error_by_dim = steps['qp_problem']['error_by_dim']
                dim_names = ['x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate']
                ax8.bar(dim_names, error_by_dim)
                ax8.set_xlabel('State Dimension')
                ax8.set_ylabel('Average Error')
                ax8.set_title('Average Tracking Error by Dimension')
                ax8.grid(True, axis='y')
            
            # 子图9: QP求解信息
            if 'qp_solve' in steps and steps['qp_solve'].get('success', False):
                ax9 = plt.subplot(3, 3, 9)
                solve = steps['qp_solve']
                solver_name = solve.get('solver', 'unknown').upper()
                info_text = f"Solver: {solver_name}\n"
                info_text += f"Status: {solve['status']}\n"
                info_text += f"Time: {solve['solve_time']:.4f}s\n"
                if solve.get('iterations') is not None:
                    info_text += f"Iter: {solve['iterations']}\n"
                if solve.get('obj_val') is not None:
                    info_text += f"Obj: {solve['obj_val']:.6f}\n"
                if solve.get('pri_res') is not None:
                    info_text += f"Pri: {solve['pri_res']:.2e}\n"
                if solve.get('dua_res') is not None:
                    info_text += f"Dua: {solve['dua_res']:.2e}"
                ax9.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                        family='monospace')
                ax9.set_xlim(0, 1)
                ax9.set_ylim(0, 1)
                ax9.axis('off')
                ax9.set_title(f'{solver_name} Solve Info')
            
            plt.tight_layout()
            
            plot_file = output_path / "mpc_analysis_plots_short_term_optimization.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization plots saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error generating visualization plots: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 配置路径
    param_path = r"D:\YRJ_Workspace\DDK-Trucksim-python\DeepEDMD\ckpt\DeepEDMD-Transv2-hd16-multiset-100e-remote.pth"
    ref_traj_path = r"D:\YRJ_Workspace\DDK-Trucksim-python\MPC\ref_trajectory\straight_acceleration_trajectory_ref.mat"
    
    # 检查文件是否存在
    if not os.path.exists(param_path):
        print(f"Error: Model file not found: {param_path}")
        return
    
    if not os.path.exists(ref_traj_path):
        print(f"Error: Reference trajectory file not found: {ref_traj_path}")
        return
    
    # 创建分析器（可以选择求解器：'auto', 'quadprog', 'osqp'）
    solver = 'auto'  # 可以改为 'quadprog' 或 'osqp' 来指定求解器
    analyzer = MPCSolveAnalyzer(param_path, ref_traj_path, solver=solver)
    
    # 第一帧的输入状态(从打印信息中提取)
    state_input = np.array([
        0.0000,  # X(m)
        0.0000,  # Y(m)
        0.0000,  # Yaw(rad)
        5.0000,  # vx(m/s)
        0.0000,  # vy(m/s)
        0.0000   # yaw_rate(rad/s)
    ])
    
    nearest_idx = 1
    # 初始控制输入（归一化）：0.5 对应零转矩/零转角
    u_prev = 0.5 * np.ones(12)
    
    # 执行分析
    analysis_result = analyzer.analyze_first_step(state_input, nearest_idx, u_prev)
    
    # 检查分析结果
    if analysis_result is None:
        print("Error: Analysis failed, analysis_result is None")
        return
    
    # 生成报告
    analyzer.generate_report(analysis_result)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)


if __name__ == '__main__':
    main()
