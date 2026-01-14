# MPC控制完整流程说明

## 概述
本文档详细说明DeepEDMD-MPC控制器从输入到输出的完整处理流程，包括所有中间步骤和数据处理。

---

## 一、初始化阶段（仅执行一次）

### 1.1 模型加载 (`DeepEDMD.__init__`)
**输入：**
- `param_path`: PyTorch模型文件路径（`.pth`格式）

**处理步骤：**
1. **文件验证**：检查文件扩展名是否为`.pth`，否则抛出`ValueError`
2. **加载PyTorch模型**：
   - 加载模型权重到指定设备（CPU/GPU）
   - 提取Encoder网络和Koopman网络
   - 设置模型为评估模式（`eval()`）
3. **提取Koopman矩阵**：
   - `A = koopman.A.cpu().numpy().T` → 形状 `(space_dim, space_dim)` = `(16, 16)`
   - `B = koopman.B.cpu().numpy().T` → 形状 `(space_dim, control_dim)` = `(16, 12)`
4. **打印Debug信息**：调用`_print_debug_info()`打印模型参数

**输出：**
- `DeepEDMD`实例，包含：
  - `encoder_model`: Transformer编码器网络
  - `koopman`: Koopman算子网络
  - `A`, `B`: Koopman矩阵（numpy格式）
  - `state_dim=6`, `control_dim=12`, `space_dim=16`, `lift_dim=10`

---

### 1.2 MPC控制器初始化 (`MPCController.__init__`)
**输入：**
- `deepedmd`: DeepEDMD模型实例
- `Np=30`: 预测时域
- `Nc=30`: 控制时域
- `Q`: 状态权重矩阵（6×6，默认对角矩阵）
- `R`: 控制权重矩阵（12×12，默认对角矩阵）
- `delta_umax`: 控制增量约束（12维）

**处理步骤：**
1. **构建扩展状态空间模型** (`_build_extended_model`)：
   - 扩展状态：`kesi = [x_lift; u_prev]`，维度 `(Nx+Nu,) = (16+12,) = (28,)`
   - 扩展系统矩阵：
     ```
     A_ext = [A  B]    形状 (28, 28)
            [0  I]
     B_ext = [B]        形状 (28, 12)
            [I]
     ```
   - 输出矩阵：`C = [I(6×6) 0(6×22)]`，只输出原始6维状态

2. **预计算预测矩阵** (`_precompute_prediction_matrices`)：
   - **PHI矩阵**（自由响应矩阵）：
     - `PHI[j] = C @ A_ext^j`，`j = 1, 2, ..., Np`
     - 形状：`(Np*state_dim, Nx+Nu) = (30*6, 28) = (180, 28)`
   - **THETA矩阵**（强制响应矩阵）：
     - `THETA[j,k] = C @ A_ext^(j-k) @ B_ext`，如果 `k <= j`，否则为0
     - 形状：`(Np*state_dim, Nc*Nu) = (180, 360)`

3. **构建约束矩阵** (`_build_constraints`)：
   - **累积约束矩阵** `A_l`：
     - 下三角矩阵，`A_l[p,q] = 1` 如果 `q <= p`，否则为0
     - 形状：`(Nc, Nc) = (30, 30)`
     - Kronecker积：`A_l = kron(A_l, I(12))`，形状 `(360, 360)`
   - **控制边界**：
     - `Umin = [0, 0, ..., 0]`（12*30维）
     - `Umax = [1, 1, ..., 1]`（12*30维）
     - `delta_Umin = -delta_umax`（12*30维）
     - `delta_Umax = delta_umax`（12*30维）
   - **预计算Hessian矩阵**：
     - `Q_kron = kron(eye(Np), Q)`，形状 `(180, 180)`
     - `R_kron = kron(eye(Nc), R)`，形状 `(360, 360)`
     - `H_11 = 2 * THETA^T @ Q_kron @ THETA + R_kron`，形状 `(360, 360)`
     - `H = [H_11  H_12]`，形状 `(361, 361)`（包含松弛变量）
            `[H_12^T H_22]`
   - **预计算约束矩阵** `A_combined`：
     - 合并不等式约束和边界约束
     - 形状：`(约束数, 361)`

4. **检查可用求解器**：检查OSQP和quadprog是否可用

**输出：**
- `MPCController`实例，包含预计算的所有矩阵

---

### 1.3 参考轨迹加载 (`initialize_controller`)
**输入：**
- `param_path`: 模型文件路径
- `data_path`: 参考轨迹数据文件路径（`.mat`格式）
- `Np`, `Nc`, `sample_interval`: MPC参数

**处理步骤：**
1. 加载`.mat`文件，提取参考轨迹数据
2. 支持三种数据格式：
   - 新格式：`ref_trajectory.position` 和 `ref_trajectory.velocity`
   - Trucksim格式：`position` 和 `velocity` 字段
   - Carsim格式：`Pos` 和 `X` 字段
3. 合并位置和速度：`ref_traj = [position, velocity]`，形状 `(N, 6)`
   - 每行：`[X(m), Y(m), Yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]`

**输出：**
- 全局状态`ref_traj`，存储在`_controller_state['ref_traj']`

---

## 二、实时控制循环（每个仿真步执行）

### 2.1 输入处理 (`compute_control`)
**输入：**
- `state_input`: 车辆当前状态 `[X, Y, Yaw, vx, vy, yaw_rate]`（6维）
  - 单位：米(m)、弧度(rad)、米/秒(m/s)、弧度/秒(rad/s)
  - 来自Trucksim，已经是国际标准单位

**处理步骤：**
1. **类型转换**：确保输入为numpy数组
2. **状态提取**：
   ```python
   x_cur = [X, Y, Yaw, vx, vy, yaw_rate]  # (6,)
   ```
3. **索引更新**：`index += 1`

---

### 2.2 参考轨迹跟踪点查找
**处理步骤：**
1. **初始化**（`index == 2`时）：
   - 查找起始最近点：`start_idx = find_nearest_point(ref_traj, x_cur, 1, len(ref_traj))`
   - 初始化：`nearest_idx = start_idx`

2. **更新最近点索引**：
   - 从当前`nearest_idx`开始向前搜索（搜索范围500个点）
   - `tmp_nearest_idx = find_nearest_point(ref_traj, x_cur, nearest_idx, 500)`
   - **重要**：索引只能向前或保持不变，不能后退
   - `new_idx = max(nearest_idx, tmp_nearest_idx)`

3. **卡住检测**：
   - 如果车辆状态变化很小（`< 0.001`），`bad_count += 1`
   - 如果`bad_count > 30`且索引不变，强制推进索引

4. **边界检查**：
   - 如果接近轨迹终点（`nearest_idx >= len(ref_traj) - Np*sample_interval - 200`），返回零控制
   - 如果车辆卡住（`bad_count > 50`），返回零控制

---

### 2.3 参考轨迹提取
**处理步骤：**
1. **提取参考轨迹段**：
   ```python
   end_idx = min(nearest_idx + Np * sample_interval, len(ref_traj))
   temp_refr = ref_traj[nearest_idx:end_idx:sample_interval, :]
   ```
   - 从`nearest_idx`开始，每隔`sample_interval`取一个点，共`Np`个点
   - 形状：`(Np, 6)`

2. **填充处理**：
   - 如果提取的点数 `< Np`，用最后一个点填充
   - 如果提取的点数 `> Np`，截断到`Np`个点

3. **参考位置提取**：
   ```python
   ref_pos_0 = temp_refr[0, :3]  # [X, Y, Yaw] 第一个参考点的位置
   ```

---

### 2.4 坐标变换和编码 (`DeepEDMD`方法)

#### 2.4.1 参考轨迹坐标变换 (`get_reference`)
**输入：**
- `temp_refr`: 全局参考轨迹 `(Np, 6)`
- `ref_pos_0`: 参考位置 `[X, Y, Yaw]`

**处理步骤：**
对每个参考点执行坐标变换：
```python
for i in range(Np):
    ref_local[i] = normalization(temp_refr[i], ref_pos_0)
```

**坐标变换公式：**
- 相对位置：`dx = X - X_ref`, `dy = Y - Y_ref`, `dyaw = Yaw - Yaw_ref`
- 旋转矩阵：`cos_yaw = cos(Yaw_ref)`, `sin_yaw = sin(Yaw_ref)`
- 局部位置：
  ```
  x_local = dx * cos_yaw + dy * sin_yaw
  y_local = -dx * sin_yaw + dy * cos_yaw
  yaw_local = dyaw
  ```
- 局部速度：直接使用全局速度（不进行坐标变换）
  ```
  vx_local = vx
  vy_local = vy
  yaw_rate_local = yaw_rate
  ```

**输出：**
- `ref_r`: 局部参考轨迹 `(Np, 6)`

---

#### 2.4.2 当前状态坐标变换 (`normalization`)
**输入：**
- `x_cur`: 全局状态 `[X, Y, Yaw, vx, vy, yaw_rate]`
- `ref_pos_0`: 参考位置 `[X, Y, Yaw]`

**处理步骤：**
- 与参考轨迹坐标变换相同

**输出：**
- `x_normalized`: 局部状态 `[x_local, y_local, yaw_local, vx_local, vy_local, yaw_rate_local]` (6维)
- **注意**：不进行数值归一化，编码器内部有BatchNorm

---

#### 2.4.3 状态编码 (`encoder`)
**输入：**
- `x_normalized`: 局部状态 `(6,)` 或 `(1, 6)`

**处理步骤：**
1. **输入预处理**：
   - 确保输入为2D数组：`(1, 6)`
   - 转换为torch tensor：`x_tensor = torch.from_numpy(x_normalized).to(device)`
   - 添加序列维度：`x_tensor = x_tensor.unsqueeze(1)` → 形状 `(1, 1, 6)`

2. **Encoder前向传播**：
   - Transformer编码器处理
   - 输出：提升状态 `(1, 1, space_dim) = (1, 1, 16)`

3. **输出处理**：
   - 移除序列维度：`x_lift = output.squeeze(1).squeeze(0)` → 形状 `(16,)`
   - 转换为numpy：`x_lift = x_lift.cpu().numpy()`

**输出：**
- `x_lift`: 提升状态 `(16,)`

---

### 2.5 MPC优化求解 (`MPCController.solve`)

#### 2.5.1 构建优化问题
**输入：**
- `x_lift`: 当前提升状态 `(16,)`
- `u_prev`: 上一时刻归一化控制输入 `(12,)`，范围 `[0, 1]`
- `ref_r`: 局部参考轨迹 `(Np, 6)`

**处理步骤：**
1. **构建扩展状态**：
   ```python
   kesi = [x_lift; u_prev]  # (28,)
   ```

2. **构建参考轨迹向量**：
   ```python
   ref_r_vec = ref_r.T.flatten('F')  # 转置后按列优先展开
   # 形状: (Np*state_dim,) = (180,)
   ```

3. **计算跟踪误差**：
   ```python
   error = PHI @ kesi - ref_r_vec  # (180,)
   ```

4. **构建QP问题**：
   - **目标函数**：`min 0.5 * x^T * H * x + f^T * x`
   - **决策变量**：`x = [delta_U; slack]`
     - `delta_U`: 控制增量序列 `(Nc*Nu,) = (360,)`
     - `slack`: 松弛变量 `(1,)`
   - **Hessian矩阵**：使用预计算的`H`（形状 `(361, 361)`）
   - **梯度向量**：
     ```python
     f_1 = 2 * error^T @ Q_kron @ THETA  # (360,)
     f_2 = [0.0]  # 松弛变量系数
     f = [f_1; f_2]  # (361,)
     ```

5. **构建约束**：
   - **不等式约束**：`A_ineq * x <= b_ineq`
     - `A_ineq = [A_l; -A_l]`，形状 `(720, 361)`
     - `Ut = kron(ones(Nc), u_prev)`，形状 `(360,)`
     - `b_ineq_1 = Umax - Ut`
     - `b_ineq_2 = -Umin + Ut`
     - `b_ineq = [b_ineq_1; b_ineq_2]`，形状 `(720,)`
   - **边界约束**：
     - `lb = [delta_Umin; 0]`，形状 `(361,)`
     - `ub = [delta_Umax; inf]`，形状 `(361,)`
     - **动态调整**：当`u_prev`接近边界时，调整`delta_U`的边界以确保约束可行

---

#### 2.5.2 QP求解
**求解器选择（按优先级）：**

1. **OSQP**（优先）：
   - 标准形式：`min 0.5 * x^T * P * x + q^T * x, s.t. l <= A*x <= u`
   - 使用预计算的稀疏矩阵
   - 支持热启动（warm start）

2. **quadprog**（备选）：
   - 标准形式：`min 0.5 * x^T * G * x + a^T * x, s.t. C^T * x >= b`
   - 需要转换约束格式
   - 使用预计算的密集矩阵

3. **失败处理**：
   - 如果所有求解器都失败，返回零控制增量

**输出：**
- `delta_u`: 控制增量 `(12,)`（只取第一个控制时域的值）
- `success`: 求解是否成功（布尔值）

---

### 2.6 控制量更新
**处理步骤：**
1. **更新归一化控制量**：
   ```python
   u_new = u_prev + delta_u  # (12,)
   u_clipped = clip(u_new, 0.0, 1.0)  # 限制在[0, 1]范围内
   u_prev = u_clipped  # 更新全局状态
   ```

2. **如果求解失败**：保持`u_prev`不变

---

### 2.7 控制输出转换 (`convert_to_control_output`)
**输入：**
- `u_normalized`: 归一化控制输入 `(12,)`，范围 `[0, 1]`

**处理步骤：**
1. **反归一化** (`denormalize_control`)：
   ```python
   control_range = control_max - control_min
   u_denorm = u_normalized * control_range + control_min
   ```
   - `control_min = [-1500, -1500, ..., -π/2, -π/2, ...]`（12维）
   - `control_max = [1500, 1500, ..., π/2, π/2, ...]`（12维）
   - 前6维：转矩范围 `[-1500, 1500]` N·m
   - 后6维：转向角范围 `[-π/2, π/2]` rad

2. **提取转矩和转向角**：
   ```python
   torques = u_denorm[:6]      # (6,) 单位：N·m
   steer_angles = u_denorm[6:]  # (6,) 单位：rad
   ```

3. **转换为Trucksim格式**：
   ```python
   steer_angles_deg = steer_angles * 180 / π  # 转换为度
   control_output = [
       steer_angles_deg[0],  # steer_LF (deg)
       steer_angles_deg[1],  # steer_RF (deg)
       steer_angles_deg[2],  # steer_LM (deg)
       steer_angles_deg[3],  # steer_RM (deg)
       steer_angles_deg[4],  # steer_LR (deg)
       steer_angles_deg[5],  # steer_RR (deg)
       torques[0],  # torque_LF (N·m)
       torques[1],  # torque_RF (N·m)
       torques[2],  # torque_LM (N·m)
       torques[3],  # torque_RM (N·m)
       torques[4],  # torque_LR (N·m)
       torques[5]   # torque_RR (N·m)
   ]
   ```

**输出：**
- `control_output`: 控制信号 `(12,)`
  - 前6维：转向角（度）
  - 后6维：转矩（N·m）

---

## 三、数据流图

```
Simulink输入
    ↓
[state_input: 6维全局状态]
    ↓
┌─────────────────────────────────────┐
│ 1. 参考轨迹跟踪点查找                │
│    - find_nearest_point()           │
│    - 更新nearest_idx                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 参考轨迹提取                      │
│    - ref_traj[nearest_idx:end:sample]│
│    - 提取Np个点                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 坐标变换                          │
│    - get_reference(): 参考轨迹       │
│    - normalization(): 当前状态        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 状态编码                          │
│    - encoder(): 6维 → 16维          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. MPC优化求解                       │
│    - 构建QP问题                      │
│    - 求解器（OSQP/quadprog）         │
│    - 输出delta_u                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. 控制量更新                        │
│    - u_new = u_prev + delta_u        │
│    - 限幅到[0, 1]                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 7. 控制输出转换                      │
│    - 反归一化                        │
│    - 单位转换（rad→deg）             │
│    - 重排为Trucksim格式              │
└─────────────────────────────────────┘
    ↓
[control_output: 12维控制信号]
    ↓
Simulink输出
```

---

## 四、关键参数说明

### 4.1 状态维度
- **原始状态** (`state_dim`): 6维
  - `[x, y, yaw, vx, vy, yaw_rate]`
- **提升状态** (`space_dim`): 16维
  - `[原始6维状态 + 10维提升特征]`

### 4.2 控制维度
- **控制输入** (`control_dim`): 12维
  - 前6维：6个车轮转矩 (N·m)
  - 后6维：6个车轮转向角 (rad)

### 4.3 MPC参数
- **预测时域** (`Np`): 30步
- **控制时域** (`Nc`): 30步
- **采样间隔** (`sample_interval`): 5（参考轨迹采样间隔）

### 4.4 归一化范围
- **控制输入归一化**：
  - 转矩：`[-1500, 1500]` N·m → `[0, 1]`
  - 转向角：`[-π/2, π/2]` rad → `[0, 1]`
- **状态归一化**：
  - 编码器内部通过BatchNorm完成，外部不做数值归一化

---

## 五、注意事项

1. **坐标变换**：
   - 所有状态都转换到以参考点为原点的局部坐标系
   - 速度分量不进行坐标变换（直接使用全局速度）

2. **索引管理**：
   - `nearest_idx`只能向前或保持不变，不能后退
   - 这确保了参考轨迹始终向前推进

3. **约束处理**：
   - 控制增量约束会根据`u_prev`动态调整，确保约束可行
   - 当`u_prev`接近边界时，限制`delta_u`的方向

4. **求解器选择**：
   - 优先使用OSQP（高性能）
   - 备选quadprog（稳定性好）
   - 如果都失败，返回零控制增量

5. **单位转换**：
   - Trucksim输入输出都是国际标准单位
   - 只有转向角需要从弧度转换为度（输出时）

---

## 六、调试信息

### 6.1 初始化时打印
- DeepEDMD模型参数（A、B矩阵、网络结构、权重统计等）

### 6.2 运行时打印（前10步或每100步）
- 车辆状态
- 参考轨迹索引
- 归一化控制量
- 控制增量
- 实际控制输出
- 求解状态

### 6.3 可选打印
- MPC跟踪误差（通过`mpc._print_tracking_error`控制）

---

## 七、错误处理

1. **控制器未初始化**：返回零控制
2. **参考轨迹未加载**：返回零控制
3. **接近轨迹终点**：返回零控制
4. **车辆卡住**：返回零控制
5. **QP求解失败**：保持上一时刻控制
6. **异常捕获**：返回零控制（避免MATLAB代码生成问题）

---

## 八、性能优化

1. **预计算矩阵**：
   - Hessian矩阵`H`
   - 预测矩阵`PHI`和`THETA`
   - 约束矩阵`A_combined`

2. **稀疏矩阵**：
   - OSQP使用稀疏矩阵格式（CSC）

3. **热启动**：
   - OSQP支持热启动，加速求解

4. **延迟加载**：
   - 求解器在第一次调用时创建

---

## 九、总结

MPC控制流程的核心是：
1. **状态处理**：全局坐标 → 局部坐标 → 提升空间
2. **优化求解**：基于Koopman线性模型求解QP问题
3. **控制输出**：归一化控制 → 反归一化 → 单位转换 → Trucksim格式

整个过程保证了：
- 实时性（预计算矩阵、高效求解器）
- 稳定性（约束处理、错误处理）
- 准确性（坐标变换、状态编码）
