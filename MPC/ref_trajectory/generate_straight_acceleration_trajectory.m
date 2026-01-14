% 生成直线加速参考轨迹
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

% 加速度参数
acceleration = (vx_final - vx0) / T_total;  % x方向加速度（m/s^2）

% 初始化数组
position = struct('x', zeros(N, 1), 'y', zeros(N, 1), 'yaw', zeros(N, 1));
velocity = struct('vx', zeros(N, 1), 'vy', zeros(N, 1), 'wz', zeros(N, 1));

% 生成直线加速轨迹
for i = 1:N
    % x方向：加速运动
    % x(t) = vx0*t + 0.5*a*t^2
    position.x(i) = vx0 * t(i) + 0.5 * acceleration * t(i)^2;
    
    % y方向：保持为0（直线运动）
    position.y(i) = 0;
    
    % 航向角yaw：保持为0（沿x轴正方向）
    position.yaw(i) = 0;
    
    % x方向速度：线性加速
    velocity.vx(i) = vx0 + acceleration * t(i);
    
    % y方向速度：保持为0（直线运动）
    velocity.vy(i) = 0;
    
    % 角速度wz：保持为0（直线运动，无转向）
    velocity.wz(i) = 0;
end

% 创建参考轨迹结构体
ref_trajectory = struct();
ref_trajectory.time = t';
ref_trajectory.position = position;
ref_trajectory.velocity = velocity;

% 保存为MAT文件
save('straight_acceleration_trajectory_ref.mat', 'ref_trajectory');

% 显示轨迹信息
fprintf('直线加速参考轨迹生成完成！\n');
fprintf('总时长: %.2f 秒\n', T_total);
fprintf('采样点数: %d\n', N);
fprintf('采样频率: %.0f Hz\n', 1/dt);
fprintf('初始速度 vx: %.2f m/s\n', vx0);
fprintf('最终速度 vx: %.2f m/s\n', velocity.vx(end));
fprintf('加速度: %.2f m/s^2\n', acceleration);
fprintf('轨迹已保存到: straight_acceleration_trajectory_ref.mat\n\n');

% 绘制轨迹图
figure('Position', [100, 100, 1200, 800]);

% 子图1：x-y平面轨迹（直线）
subplot(2, 3, 1);
plot(position.x, position.y, 'b-', 'LineWidth', 2);
xlabel('X (m)');
ylabel('Y (m)');
title('直线轨迹 (X-Y平面)');
grid on;
axis equal;
xlim([0, max(position.x) * 1.1]);

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
ylim([-1, 1]);

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
ylim([-1, 1]);

% 子图6：合速度随时间变化（等于vx，因为vy=0）
subplot(2, 3, 6);
v_total = sqrt(velocity.vx.^2 + velocity.vy.^2);
plot(t, v_total, 'k-', 'LineWidth', 1.5); hold on;
plot(t, velocity.vx, 'r--', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('速度 (m/s)');
title('合速度随时间变化');
legend('合速度', 'vx', 'Location', 'best');
grid on;

sgtitle('直线加速参考轨迹分析', 'FontSize', 14, 'FontWeight', 'bold');

% 显示前10个数据点
fprintf('前10个参考轨迹点:\n');
fprintf('时间(s)\t\tX(m)\t\tY(m)\t\tYaw(deg)\tvx(m/s)\t\tvy(m/s)\twz(deg/s)\n');
fprintf('----------------------------------------------------------------------------\n');
for i = 1:min(10, N)
    fprintf('%.2f\t\t%.4f\t\t%.4f\t\t%.2f\t\t%.4f\t\t%.4f\t\t%.2f\n', ...
        t(i), position.x(i), position.y(i), position.yaw(i)*180/pi, ...
        velocity.vx(i), velocity.vy(i), velocity.wz(i)*180/pi);
end

% 显示最后10个数据点
fprintf('\n最后10个参考轨迹点:\n');
fprintf('时间(s)\t\tX(m)\t\tY(m)\t\tYaw(deg)\tvx(m/s)\t\tvy(m/s)\twz(deg/s)\n');
fprintf('----------------------------------------------------------------------------\n');
for i = max(1, N-9):N
    fprintf('%.2f\t\t%.4f\t\t%.4f\t\t%.2f\t\t%.4f\t\t%.4f\t\t%.2f\n', ...
        t(i), position.x(i), position.y(i), position.yaw(i)*180/pi, ...
        velocity.vx(i), velocity.vy(i), velocity.wz(i)*180/pi);
end

fprintf('\n轨迹生成完成！\n');
