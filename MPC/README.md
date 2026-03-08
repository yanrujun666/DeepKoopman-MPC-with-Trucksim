# MPC 模块说明

本目录为 Koopman-MPC 与 Simulink/Trucksim 闭环仿真对接的 Python 端实现。

## 调用关系（当前实际走的链路）

```
Simulink (1 ms)
    → MATLAB Function（matlab_function_wrapper.m）
    → py.ddk_mpc_sfunction.initialize_controller / compute_control / reset_controller
    → koopman_mpc_v2：load_model、KoopmanMPC、encode_state、encode_ref_trajectory、convert_to_control_output
    → 控制输出 12 维 → Trucksim
```

**结论**：闭环仅使用 **koopman_mpc_v2**；`ddk_mpc_sfunction.py` 只导入 `koopman_mpc_v2`，不依赖其他 MPC 实现。

## 核心文件（Simulink 闭环）

| 文件 | 说明 |
|------|------|
| **ddk_mpc_sfunction.py** | Simulink 调用的 Python 接口：`initialize_controller`、`compute_control`、`reset_controller`，仅使用 koopman_mpc_v2。 |
| **koopman_mpc_v2.py** | 新网络 + 新 MPC：WithoutNorm 编码器、Koopmanv1、KoopmanMPC（12 维控制、硬约束）、加载/编码/输出转换。 |
| **matlab_function_wrapper.m** | MATLAB Function 块内代码，配置 ckpt 与参考轨迹路径并调用上述 Python 接口。 |

更多文件作用与调用关系见 **markdown/代码结构说明.md**。

## 目录

- **ref_trajectory/**：参考轨迹生成脚本（.m）；`test_structure.py` 为可选调试脚本，用于检查 .mat 结构。
- **markdown/**：适配说明、控制流程、代码结构等文档。

## 已移除（无引用、已废弃）

- `ddk_controller.py`：旧 Koopman-MPC（DeepEDMD + MPCController），已无调用，已删除。
- `legacy_matlab_model.py`：遗留 MATLAB 模型加载，已废弃。
- `mpc_main.py`：原批量测试主程序，已由 Simulink + ddk_mpc_sfunction 替代。
- `mpc_solve_analysis.py`：MPC 求解分析脚本，无其他模块引用。

## 环境要求与依赖

### 必须的库（与代码一一对应）

| 库 | 用途 | 使用位置 |
|----|------|----------|
| **numpy** (且须 **&lt;2**) | 数组、与 PyTorch 互转 | 全链路；PyTorch 在 NumPy 2.x 下会报 `Numpy is not available` |
| **torch** | 加载 .pth、编码器前向、Koopman 矩阵 | `koopman_mpc_v2`：load_model、encode_state、encode_ref_trajectory、KoopmanMPC |
| **scipy** | 读参考轨迹 .mat（`loadmat`） | `ddk_mpc_sfunction.initialize_controller` |
| **cvxpy** | MPC QP 求解（OSQP） | `koopman_mpc_v2.KoopmanMPC._solve_mpc_qp_single` |

### 安装方式

- **Conda（推荐，与 MATLAB 共用同一环境）**  
  使用本目录的 `environment.yml` 创建环境：
  ```bash
  conda env create -f MPC/environment.yml
  conda activate mpc-control
  ```
  再在 MATLAB 中指定该环境的 Python 解释器。

- **pip（已有环境）**  
  在 MATLAB 使用的 Python 环境中执行：
  ```bash
  pip install -r MPC/requirements.txt
  ```
  若当前为 NumPy 2.x，可仅降级 NumPy：
  ```bash
  pip install "numpy>=1.20,<2"
  ```

## 使用流程

1. 在 Simulink 中配置 Python 路径（如 `py.sys.path().insert(int32(0), '...\MPC')`）。
2. MATLAB Function 块使用 `matlab_function_wrapper.m` 中的逻辑，指定 `param_path`（.pth）和 `data_path`（参考轨迹 .mat）。
3. 仿真时每次调用 `compute_control(state_6)`，返回 12 维控制（6 转向角 deg + 6 转矩 N·m）。
