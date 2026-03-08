"""
读取 PyTorch 权重文件的脚本
读取 LinearModel-multiset.pth 文件并显示其内容
"""

import torch
import os
import sys

# 设置输出编码为 UTF-8（Windows 系统）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def read_weights(weights_path):
    """
    读取 PyTorch 权重文件
    
    Args:
        weights_path: 权重文件路径
    """
    if not os.path.exists(weights_path):
        print(f"错误: 文件 {weights_path} 不存在")
        return None
    
    print(f"正在读取权重文件: {weights_path}")
    print("-" * 60)
    
    try:
        # 加载权重文件
        weights = torch.load(weights_path, map_location='cpu')
        
        print(f"权重文件类型: {type(weights)}")
        print("-" * 60)
        
        # 如果是字典类型（最常见的情况）
        if isinstance(weights, dict):
            print(f"权重字典包含 {len(weights)} 个键:")
            print()
            
            for key, value in weights.items():
                print(f"键名: {key}")
                if isinstance(value, torch.Tensor):
                    print(f"  类型: torch.Tensor")
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  最小值: {value.min().item():.6f}")
                    print(f"  最大值: {value.max().item():.6f}")
                    print(f"  均值: {value.mean().item():.6f}")
                    print(f"  标准差: {value.std().item():.6f}")
                    # 如果矩阵不太大，显示前几行
                    if value.numel() <= 100:
                        print(f"  完整矩阵:\n{value}")
                    elif len(value.shape) == 2 and value.shape[0] <= 20:
                        print(f"  矩阵内容:\n{value}")
                elif isinstance(value, (int, float)):
                    print(f"  值: {value}")
                elif isinstance(value, str):
                    print(f"  值: {value}")
                elif hasattr(value, 'keys'):  # OrderedDict 或其他字典类型
                    print(f"  类型: {type(value).__name__}")
                    print(f"  包含 {len(value)} 个子键:")
                    for sub_key, sub_value in value.items():
                        print(f"    子键 '{sub_key}':")
                        if isinstance(sub_value, torch.Tensor):
                            print(f"      形状: {sub_value.shape}, 数据类型: {sub_value.dtype}")
                            print(f"      最小值: {sub_value.min().item():.6f}")
                            print(f"      最大值: {sub_value.max().item():.6f}")
                            print(f"      均值: {sub_value.mean().item():.6f}")
                            print(f"      标准差: {sub_value.std().item():.6f}")
                            # 显示矩阵内容：如果元素数量不太多，或者行数不太多，就显示完整内容
                            if sub_value.numel() <= 200:  # 增加阈值到200
                                print(f"      完整矩阵内容:\n{sub_value}")
                            elif len(sub_value.shape) == 2 and sub_value.shape[0] <= 30:  # 2D矩阵且行数<=30
                                print(f"      完整矩阵内容:\n{sub_value}")
                            elif len(sub_value.shape) == 2:  # 2D矩阵但行数较多，至少显示前20行
                                print(f"      矩阵内容（显示前20行）:\n{sub_value[:20]}")
                                if sub_value.shape[0] > 20:
                                    print(f"      ... (还有 {sub_value.shape[0] - 20} 行未显示)")
                            else:
                                print(f"      矩阵内容（元素较多，仅显示统计信息）")
                else:
                    print(f"  类型: {type(value)}")
                    if isinstance(value, dict):
                        print(f"  包含 {len(value)} 个子项")
                        # 只显示前几个子项
                        for i, (sub_key, sub_value) in enumerate(value.items()):
                            if i < 3:
                                print(f"    '{sub_key}': {sub_value}")
                            else:
                                print(f"    ... (还有 {len(value) - 3} 个子项)")
                                break
                    else:
                        print(f"  值: {value}")
                print()
        
        # 如果是 OrderedDict 类型
        elif hasattr(weights, 'keys'):
            print(f"权重包含 {len(weights)} 个键:")
            print()
            for key in weights.keys():
                value = weights[key]
                print(f"键名: {key}")
                if isinstance(value, torch.Tensor):
                    print(f"  类型: torch.Tensor")
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  最小值: {value.min().item():.6f}")
                    print(f"  最大值: {value.max().item():.6f}")
                    print(f"  均值: {value.mean().item():.6f}")
                    print(f"  标准差: {value.std().item():.6f}")
                else:
                    print(f"  类型: {type(value)}")
                    print(f"  值: {value}")
                print()
        
        # 如果是单个 Tensor
        elif isinstance(weights, torch.Tensor):
            print("权重是一个单独的 Tensor:")
            print(f"  形状: {weights.shape}")
            print(f"  数据类型: {weights.dtype}")
            print(f"  最小值: {weights.min().item():.6f}")
            print(f"  最大值: {weights.max().item():.6f}")
            print(f"  均值: {weights.mean().item():.6f}")
            print(f"  标准差: {weights.std().item():.6f}")
        
        else:
            print(f"权重类型: {type(weights)}")
            print(f"权重内容: {weights}")
        
        return weights
        
    except Exception as e:
        print(f"读取权重文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 权重文件路径
    weights_file = os.path.join(script_dir, "LinearModel-multiset.pth")
    
    # 读取权重文件
    weights = read_weights(weights_file)
    
    if weights is not None:
        print("-" * 60)
        print("权重文件读取完成！")
