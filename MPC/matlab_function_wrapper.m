% MATLAB Function模块包装器
% 用于在Simulink中调用Python MPC控制器
% 支持PyTorch模型（.pth格式）和MATLAB模型（.mat/.pkl格式）
%
% 使用方法：
% 1. 在Simulink中添加"MATLAB Function"块
% 2. 将此代码复制到MATLAB Function块中
% 3. 配置输入输出端口
% 4. 在Model Callbacks的InitFcn中添加Python路径：
%    py.sys.path().insert(int32(0), 'D:\YRJ_Workspace\DDK-Trucksim-python\MPC');

function control_output = ddk_mpc_matlab_function(state_input)
%#codegen
% DeepEDMD-MPC控制器MATLAB Function包装器（Trucksim版本）
% 支持12维控制输出（6个转向角 + 6个转矩）
% 仅支持.pth格式模型文件
%
% 输入:
%   state_input: [6x1] 车辆状态 [X(m), Y(m), Yaw(rad), vx(m/s), vy(m/s), yaw_rate(rad/s)]
%                Trucksim输出已经是国际标准单位，无需转换
%
% 输出:
%   control_output: [12x1] 控制信号
%      [steer_LF(deg), steer_RF(deg), steer_LM(deg), steer_RM(deg), 
%       steer_LR(deg), steer_RR(deg),
%       torque_LF(N·m), torque_RF(N·m), torque_LM(N·m), torque_RM(N·m),
%       torque_LR(N·m), torque_RR(N·m)]

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
    
    % 设置路径（根据实际情况修改）
    % 仅支持PyTorch模型（.pth格式），使用Transformer编码器
    % 示例：'D:\YRJ_Workspace\DDK-Trucksim-python\DeepEDMD\ckpt\DeepEDMD-Transv2-hd16-multiset-100e.pth'
    
    % TODO: 更新为Trucksim参数文件路径（必须是.pth格式）
    param_path = 'D:\YRJ_Workspace\DDK-Trucksim-python\DeepEDMD\ckpt\DeepEDMD-Transv2-hd16-multiset-100e-remote.pth';
    
    % TODO: 更新为Trucksim参考轨迹数据文件路径
    % 数据文件应包含 'position' 和 'velocity' 字段（或 'Pos' 和 'X' 字段）
    % data_path = 'D:\YRJ_Workspace\DDK-Trucksim-python\MPC\ref_trajectory\snake_trajectory_ref.mat';
    data_path = 'D:\YRJ_Workspace\DDK-Trucksim-python\MPC\ref_trajectory\straight_acceleration_trajectory_ref.mat';
    
    % 初始化Python控制器
    % 参数：param_path, data_path, Np=30, Nc=30, sample_interval=5
    % Np: 预测时域（步数）
    % Nc: 控制时域（步数）
    % sample_interval: 采样间隔（用于参考轨迹提取）
    py.ddk_mpc_sfunction.initialize_controller(param_path, data_path, 30, 30, 5);
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

