"""
DDK-MPC主控制程序
替代MATLAB的DDK_mpc_main_DCSUV_50ms.m
"""

import numpy as np
import scipy.io as scio
import os
import sys
from pathlib import Path
from typing import Optional
import time

# 导入控制器
from ddk_controller import DDK, MPCController

# 注意：此文件主要用于批量测试，实际Simulink集成请使用 ddk_mpc_sfunction.py


class MPCMain:
    """MPC主控制类"""
    
    def __init__(self, root_path: str, param_path: str):
        """
        初始化MPC主程序
        
        Args:
            root_path: 项目根路径
            param_path: DDK模型参数文件路径
        """
        self.root_path = Path(root_path)
        self.param_path = param_path
        
        # 配置参数
        self.sys_interval = 50  # 采样间隔50ms
        self.sample_interval = int(np.ceil(self.sys_interval / 10))  # 采样间隔（10ms基准）
        self.Np = 30  # 预测时域
        self.Nc = 30  # 控制时域
        
        # 数据路径
        # TODO: 更新为Trucksim数据集路径
        self.data_path = self.root_path / "TrucksimDatasets"  # Trucksim数据集路径
        
        # 数据集数量
        self.train_file_num = 30
        self.val_file_num = 4
        self.test_file_num = 4
        
        # MPC权重矩阵
        self.tempQ = np.diag([20, 1000, 1000, 1000, 20, 20])  # 状态权重（6维，保持不变）
        # Trucksim控制维度为12维（6个转矩+6个转向角）
        # R矩阵：转矩权重较小（允许较大变化），转向角权重较大（限制变化）
        self.RTimes = np.diag([5, 5, 5, 5, 5, 5, 10000, 10000, 10000, 10000, 10000, 10000])
        # delta_umax：12维控制增量约束
        # 转矩增量：0.5（归一化后），转向角增量：0.2（归一化后）
        self.delta_umax = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        
        # 初始化DDK模型
        print(f"加载DDK模型: {param_path}")
        self.ddk = DDK(param_path)
        
        # 初始化MPC控制器
        self.mpc = MPCController(
            self.ddk, 
            Np=self.Np, 
            Nc=self.Nc,
            Q=self.tempQ,
            R=self.RTimes,
            delta_umax=self.delta_umax
        )
        
        # 注意：如需通过MATLAB Engine运行Simulink，请自行实现
    
    def load_reference_data(self, datatype: str, idx: int) -> dict:
        """
        加载参考轨迹数据
        支持三种格式：
        1. 新格式：ref_trajectory结构体（包含position和velocity子结构体）
        2. Trucksim格式：'position' 和 'velocity' 字段（N×3数组）
        3. Carsim格式：'Pos' 和 'X' 字段（N×3数组）
        
        Args:
            datatype: 'train', 'val', 'test'
            idx: 数据文件索引
        
        Returns:
            包含Pos, X, U的字典
        """
        # TODO: 更新为Trucksim数据文件命名格式
        if datatype == 'train':
            filename = f"Trucksim_{idx}.mat"  # 需要根据实际命名格式修改
        elif datatype == 'test':
            filename = f"Trucksim_test_{idx}.mat"  # 需要根据实际命名格式修改
        elif datatype == 'val':
            filename = f"Trucksim_val_{idx}.mat"  # 需要根据实际命名格式修改
        else:
            raise ValueError(f"未知的数据类型: {datatype}")
        
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        data = scio.loadmat(str(filepath), squeeze_me=True)
        
        # 辅助函数：从MATLAB结构体中提取字段
        def extract_struct_field(struct_data, field_name):
            """从MATLAB结构体中提取字段值"""
            if isinstance(struct_data, np.ndarray) and struct_data.dtype.names:
                if struct_data.shape == () and field_name in struct_data.dtype.names:
                    value = struct_data[field_name]
                    if isinstance(value, np.ndarray) and value.ndim == 0:
                        return value.item()
                    return value
            elif hasattr(struct_data, field_name):
                return getattr(struct_data, field_name)
            return None
        
        # 检查是否是新格式（ref_trajectory结构体）
        if 'ref_trajectory' in data:
            ref_traj_struct = data['ref_trajectory']
            
            # 提取position结构体
            pos_struct = extract_struct_field(ref_traj_struct, 'position')
            pos_x = extract_struct_field(pos_struct, 'x')
            pos_y = extract_struct_field(pos_struct, 'y')
            pos_yaw = extract_struct_field(pos_struct, 'yaw')
            
            # 提取velocity结构体
            vel_struct = extract_struct_field(ref_traj_struct, 'velocity')
            vel_vx = extract_struct_field(vel_struct, 'vx')
            vel_vy = extract_struct_field(vel_struct, 'vy')
            vel_wz = extract_struct_field(vel_struct, 'wz')
            
            # 组合为数组格式
            pos = np.column_stack([np.array(pos_x).flatten(), 
                                   np.array(pos_y).flatten(), 
                                   np.array(pos_yaw).flatten()])
            vel = np.column_stack([np.array(vel_vx).flatten(), 
                                   np.array(vel_vy).flatten(), 
                                   np.array(vel_wz).flatten()])
            
            return {
                'Pos': pos,
                'X': vel,
                'U': data.get('U', None)
            }
        
        # 兼容旧格式
        return {
            'Pos': data.get('position', data.get('Pos', None)),  # 兼容两种字段名
            'X': data.get('velocity', data.get('X', None)),      # 兼容两种字段名
            'U': data.get('U', None)
        }
    
    def run_simulation(self, datatype: str, idx: int, sim_time: Optional[float] = None, 
                      is_save: bool = True) -> dict:
        """
        运行单次仿真
        
        Args:
            datatype: 数据类型
            idx: 数据索引
            sim_time: 仿真时间（秒），如果为None则使用数据长度
            is_save: 是否保存结果
        
        Returns:
            仿真结果字典
        """
        print(f"\n运行仿真: {datatype}, idx={idx}")
        
        # 加载参考数据
        ref_data = self.load_reference_data(datatype, idx)
        ref_pos = ref_data['Pos']
        ref_x = ref_data['X']
        ref_u = ref_data['U']
        
        # 计算仿真时间
        if sim_time is None:
            sim_time = np.floor(ref_pos.shape[0] / (1000 / self.sys_interval)) - 2
            sim_time = min(sim_time, 1000)  # 最大1000秒
        
        print(f"仿真时间: {sim_time}秒")
        
        # 反归一化参考数据（用于对比）
        X_max = np.array([20, 0.5, 0.5])
        X_min = np.array([0, -0.5, -0.5])
        ref_x_denorm = (ref_x + 1) * (X_max - X_min) / 2 + X_min
        ref_full = np.hstack([ref_pos, ref_x_denorm])
        
        # 注意：如需运行Simulink仿真，请直接在Simulink中使用 ddk_mpc_sfunction.py
        # 此文件主要用于数据加载和结果处理
        print("提示: 请直接在Simulink中使用 ddk_mpc_sfunction.py 进行仿真")
        return None
    
    def plot_result(self, result: dict, save_path: Optional[str] = None):
        """
        绘制结果（简化版，完整版需要实现plot_result函数）
        
        Args:
            result: 仿真结果
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            x_pred = result['x_pred']
            ref_full = result['ref_full']
            mpc_data = result['mpc_data']
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 位置跟踪
            axes[0, 0].plot(ref_full[:, 0], ref_full[:, 1], 'b-', label='参考轨迹')
            axes[0, 0].plot(x_pred[:, 0], x_pred[:, 1], 'r--', label='实际轨迹')
            axes[0, 0].set_xlabel('X (m)')
            axes[0, 0].set_ylabel('Y (m)')
            axes[0, 0].set_title('位置跟踪')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 速度跟踪
            axes[0, 1].plot(ref_full[:, 3], 'b-', label='参考vx')
            axes[0, 1].plot(x_pred[:, 3], 'r--', label='实际vx')
            axes[0, 1].set_xlabel('时间步')
            axes[0, 1].set_ylabel('速度 (m/s)')
            axes[0, 1].set_title('速度跟踪')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 控制输入
            axes[1, 0].plot(mpc_data[:, 0], label='转向角')
            axes[1, 0].set_xlabel('时间步')
            axes[1, 0].set_ylabel('角度 (deg)')
            axes[1, 0].set_title('转向角')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(mpc_data[:, 2], label='油门')
            axes[1, 1].plot(mpc_data[:, 3], label='刹车')
            axes[1, 1].set_xlabel('时间步')
            axes[1, 1].set_ylabel('控制量')
            axes[1, 1].set_title('油门/刹车')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"结果已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib未安装，无法绘图")
        except Exception as e:
            print(f"绘图失败: {e}")
    
    def run_all(self, datatype: str = 'all', is_save: bool = True):
        """
        运行所有数据集
        
        Args:
            datatype: 'all', 'train', 'val', 'test'
            is_save: 是否保存结果
        """
        # 确定要运行的数据集
        if datatype == 'all':
            total_num = self.train_file_num + self.val_file_num + self.test_file_num
            start_idx = 0
        elif datatype == 'train':
            total_num = self.train_file_num
            start_idx = 0
        elif datatype == 'val':
            total_num = self.val_file_num
            start_idx = self.train_file_num
        elif datatype == 'test':
            total_num = self.test_file_num
            start_idx = self.train_file_num + self.val_file_num
        else:
            raise ValueError(f"未知的数据类型: {datatype}")
        
        # 结果保存目录
        result_dir = self.root_path / "result"
        result_dir.mkdir(exist_ok=True)
        
        # 运行所有数据集
        for i in range(total_num):
            # 确定数据类型和索引
            if i < self.train_file_num:
                current_type = 'train'
                current_idx = i
            elif i < self.train_file_num + self.val_file_num:
                current_type = 'val'
                current_idx = i - self.train_file_num
            else:
                current_type = 'test'
                current_idx = i - self.train_file_num - self.val_file_num
            
            # 运行仿真
            result = self.run_simulation(current_type, current_idx, is_save=is_save)
            
            if result is not None and is_save:
                # 保存结果
                save_path = result_dir / f"{current_type}_{current_idx}_result.png"
                self.plot_result(result, str(save_path))
    
    def close(self):
        """清理资源"""
        pass


def main():
    """主函数"""
    # 配置路径（需要根据实际情况修改）
    root_path = r"D:\YRJ_Workspace\DDK-Trucksim-python\MPC"
    # TODO: 更新为Trucksim参数文件路径（待DDK模型训练完成后）
    param_path = r"D:\YRJ_Workspace\DDK-Trucksim-python\MPC\params_for_matlab_Trucksim.mat"  # 待定
    
    # 检查文件是否存在
    if not os.path.exists(param_path):
        print(f"错误: 参数文件不存在: {param_path}")
        print("请先运行训练脚本生成参数文件")
        return
    
    # 创建MPC主程序
    mpc_main = MPCMain(root_path, param_path)
    
    try:
        # 运行所有测试数据
        print("开始运行MPC控制...")
        mpc_main.run_all(datatype='test', is_save=True)
        print("所有仿真完成")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mpc_main.close()


if __name__ == '__main__':
    main()

