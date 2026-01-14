% 生成蛇形参考轨迹
% 包含position和velocity字段
% position: x, y, yaw
% velocity: vx, vy, wz
% 总时长25s，采样间隔0.01s，初速度vx=5m/s

clear; clc; close all;

% 参数设置
T_total = 25;           % 总时长（秒）
dt = 0.01;              % 采样间隔（秒）
t = 0:dt:T_total;       % 时间向量
N = length(t);          % 采样点数

% 初始速度
vx0 = 5;                % 初始x方向速度（m/s）
vx_final = 15;          % 最终x方向速度（m/s）

% 蛇形轨迹参数
A = 3;                  % y方向振幅（米）
omega = 0.4;            % 蛇形频率（rad/s）
acceleration = (vx_final - vx0) / T_total;  % x方向加速度（m/s^2）

% 初始化数组
position = struct('x', zeros(N, 1), 'y', zeros(N, 1), 'yaw', zeros(N, 1));
velocity = struct('vx', zeros(N, 1), 'vy', zeros(N, 1), 'wz', zeros(N, 1));

% 生成轨迹
for i = 1:N
    % x方向：加速运动
    % x(t) = vx0*t + 0.5*a*t^2
    position.x(i) = vx0 * t(i) + 0.5 * acceleration * t(i)^2;
    
    % y方向：蛇形（正弦波）
    % y(t) = A * sin(omega * t)
    position.y(i) = A * sin(omega * t(i));
    
    % x方向速度：线性加速
    velocity.vx(i) = vx0 + acceleration * t(i);
    
    % y方向速度：对y求导
    % vy = dy/dt = A * omega * cos(omega * t)
    velocity.vy(i) = A * omega * cos(omega * t(i));
    
    % 航向角yaw：根据速度方向计算
    % yaw = atan2(vy, vx)
    position.yaw(i) = atan2(velocity.vy(i), velocity.vx(i));
    
    % 角速度wz：对yaw求导
    if i > 1
        % 使用数值微分计算角速度
        dyaw = position.yaw(i) - position.yaw(i-1);
        % 处理角度跳变（-pi到pi）
        if dyaw > pi
            dyaw = dyaw - 2*pi;
        elseif dyaw < -pi
            dyaw = dyaw + 2*pi;
        end
        velocity.wz(i) = dyaw / dt;
    else
        velocity.wz(i) = 0;
    end
end

% 更精确的wz计算（使用解析方法）
% wz = d(yaw)/dt = d(atan2(vy, vx))/dt
% 使用导数公式：d(atan2(y,x))/dt = (x*dy/dt - y*dx/dt) / (x^2 + y^2)
for i = 1:N
    vx = velocity.vx(i);
    vy = velocity.vy(i);
    % vy对t的导数：d(vy)/dt = -A*omega^2*sin(omega*t)
    dvy_dt = -A * omega^2 * sin(omega * t(i));
    % vx对t的导数：d(vx)/dt = acceleration
    dvx_dt = acceleration;
    
    % 计算wz
    if abs(vx) > 1e-6 || abs(vy) > 1e-6
        velocity.wz(i) = (vx * dvy_dt - vy * dvx_dt) / (vx^2 + vy^2);
    else
        velocity.wz(i) = 0;
    end
end

% 创建参考轨迹结构体
ref_trajectory = struct();
ref_trajectory.time = t';
ref_trajectory.position = position;
ref_trajectory.velocity = velocity;

% 保存为MAT文件
save('snake_trajectory_ref.mat', 'ref_trajectory');

% 显示轨迹信息
fprintf('参考轨迹生成完成！\n');
fprintf('总时长: %.2f 秒\n', T_total);
fprintf('采样点数: %d\n', N);
fprintf('采样频率: %.0f Hz\n', 1/dt);
fprintf('初始速度 vx: %.2f m/s\n', vx0);
fprintf('最终速度 vx: %.2f m/s\n', velocity.vx(end));
fprintf('加速度: %.2f m/s^2\n', acceleration);
fprintf('轨迹已保存到: snake_trajectory_ref.mat\n\n');

% 绘制轨迹图
figure('Position', [100, 100, 1200, 800]);

% 子图1：x-y平面轨迹
subplot(2, 3, 1);
plot(position.x, position.y, 'b-', 'LineWidth', 1.5);
xlabel('X (m)');
ylabel('Y (m)');
title('蛇形轨迹 (X-Y平面)');
grid on;
axis equal;

% 子图2：位置随时间变化
subplot(2, 3, 2);
plot(t, position.x, 'r-', 'LineWidth', 1.5); hold on;
plot(t, position.y, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('位置 (m)');
title('位置随时间变化');
legend('X', 'Y', 'Location', 'best');
grid on;

% 子图3：航向角随时间变化
subplot(2, 3, 3);
plot(t, position.yaw * 180/pi, 'g-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('航向角 (度)');
title('航向角随时间变化');
grid on;

% 子图4：速度随时间变化
subplot(2, 3, 4);
plot(t, velocity.vx, 'r-', 'LineWidth', 1.5); hold on;
plot(t, velocity.vy, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('速度 (m/s)');
title('速度随时间变化');
legend('vx', 'vy', 'Location', 'best');
grid on;

% 子图5：角速度随时间变化
subplot(2, 3, 5);
plot(t, velocity.wz * 180/pi, 'm-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('角速度 (度/s)');
title('角速度随时间变化');
grid on;

% 子图6：合速度随时间变化
subplot(2, 3, 6);
v_total = sqrt(velocity.vx.^2 + velocity.vy.^2);
plot(t, v_total, 'k-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('合速度 (m/s)');
title('合速度随时间变化');
grid on;

sgtitle('蛇形参考轨迹分析', 'FontSize', 14, 'FontWeight', 'bold');

% 显示前10个数据点
fprintf('前10个参考轨迹点:\n');
fprintf('时间(s)\t\tX(m)\t\tY(m)\t\tYaw(deg)\tvx(m/s)\t\tvy(m/s)\twz(deg/s)\n');
fprintf('----------------------------------------------------------------------------\n');
for i = 1:min(10, N)
    fprintf('%.2f\t\t%.4f\t\t%.4f\t\t%.2f\t\t%.4f\t\t%.4f\t\t%.2f\n', ...
        t(i), position.x(i), position.y(i), position.yaw(i)*180/pi, ...
        velocity.vx(i), velocity.vy(i), velocity.wz(i)*180/pi);
end

fprintf('\n轨迹生成完成！\n');
