% MATLAB Function模块包装器
% 用于在Simulink中调用Python MPC控制器（Koopman-MPC V2：新网络 + 新MPC）
%
% 使用方法：
% 1. 在Simulink中添加"MATLAB Function"块
% 2. 将此代码复制到MATLAB Function块中
% 3. 配置输入输出端口
% 4. 在Model Callbacks的InitFcn中添加Python路径：
%    py.sys.path().insert(int32(0), 'D:\YRJ_Workspace\DDK-Trucksim-python\MPC');

function control_output = ddk_mpc_matlab_function(state_input)
%#codegen
% Koopman-MPC V2 控制器 MATLAB Function 包装器（Trucksim）
% 新 ckpt：CustomEncoderUnscaledv2WithoutNorm + KoopmanMPC（12维控制、硬约束）
%
% 输入:
%   state_input: [6x1] 车辆状态（相对起点）[X(m), Y(m), Yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]
%
% 输出:
%   control_output: [12x1] [steer_LF..RR(deg), torque_LF..RR(N·m)]

% 显式声明输入输出大小（帮助代码生成器确定维度）
assert(isequal(size(state_input), [6, 1]), 'state_input must be 6x1');
control_output = zeros(12, 1);  % 预分配输出大小（Trucksim为12维）

% 声明Python函数为外部函数（不参与代码生成）
coder.extrinsic('py.ddk_mpc_sfunction.initialize_controller');
coder.extrinsic('py.ddk_mpc_sfunction.compute_control');
coder.extrinsic('py.ddk_mpc_sfunction.reset_controller');

% 持久变量（在仿真过程中保持状态）
persistent is_initialized;
persistent param_path;
persistent data_path;

% 初始化（第一次调用时）
% 注意：Python路径配置必须在Simulink运行前在MATLAB命令窗口执行：
% py.sys.path().insert(int32(0), 'D:\YRJ_Workspace\DDK-Trucksim-python\MPC');
% 或者在Model Callbacks的InitFcn中添加上述命令
if isempty(is_initialized)
    is_initialized = false;
    
    % 新 ckpt 路径（与 mpc_dk 一致，项目根下 ckpt 文件夹）
    param_path = 'D:\YRJ_Workspace\DDK-Trucksim-python\ckpt\DeepEDMD-Transv2wonorm-hd16-multiset-100e-remote-local-lr1e-4-rollover-0.05pilossv24-0222.pth';
    
    % 参考轨迹 .mat（支持 ref_trajectory / position+velocity / Pos+X 格式）
    data_path = 'D:\YRJ_Workspace\DDK-Trucksim-python\data\ref_traj\all\all_wheel_steer_Scenario_snake_acc_5m_s_ref.mat';
    
    % ---------- Q / R / I 超参数（可选，传入 Python 用于 MPC 代价权重）----------
    % Q: 状态/跟踪权重 [16]，前 6 维为物理状态 (x,y,yaw,vx,vy,wz)，后 10 维为 Koopman 提升维
    Q = [ones(6,1); ones(10,1)];   % 默认全 1；可改为标量或逐维
    % R: 控制偏离中性(0.5)的惩罚 [12]，前 6 转矩、后 6 转向
    R = 0.01 * ones(12, 1);        % 默认 0.01；增大可更保守
    % I: 控制增量/平滑惩罚 [12]
    I = 1.5 * ones(12, 1);         % 默认 0.5；增大可减轻抖动
    % 若希望使用 Python 端默认值，可将 Q、R、I 任一设为 []，例如: Q=[]; R=[]; I=[];
    
    % 初始化 Koopman-MPC V2：param_path, data_path, Np, Nc, sample_interval, decimation, Q, R, I
    % sample_interval=1 与 Koopman 0.01s 对齐；decimation=10 表示每 10 次调用求解一次 MPC
    py.ddk_mpc_sfunction.initialize_controller(param_path, data_path, 30, 30, 1, 10, Q, R, I);
    % 为避免Python侧全局状态在多次仿真间残留，初始化后显式重置一次状态
    % 重要：Python侧u_prev为“归一化控制”，应以0.5作为零转矩/零转角的初始值
    py.ddk_mpc_sfunction.reset_controller();
    is_initialized = true;
end

% 调用Python MPC控制器
% compute_control返回12维控制输出：
% [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
%  steer_LR(deg), steer_RR(deg),
%  torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
%  torque_LR(N·m), torque_RR(N·m)]
control_py = py.ddk_mpc_sfunction.compute_control(state_input);
control_output = double(control_py);
control_output = reshape(control_output(:), 12, 1);  % 确保输出为12x1

% 安全检查：确保输出维度正确
if numel(control_output) ~= 12
    warning('Control output dimension error, returning zero control');
    control_output = zeros(12, 1);
end

% 可选：添加控制输出限幅（如果需要）
% 转向角限制：±30度
control_output(1:6) = max(-30, min(30, control_output(1:6)));
% 转矩限制：根据实际车辆参数设置
control_output(7:12) = max(-1000, min(1000, control_output(7:12)));

end

