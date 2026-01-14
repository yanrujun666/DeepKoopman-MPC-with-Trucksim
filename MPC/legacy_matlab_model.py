"""
遗留的MATLAB模型支持代码
此文件包含已废弃的MATLAB模型（.mat/.pkl格式）加载和处理方法
这些方法已不再使用，仅保留用于参考或未来可能的恢复
"""

import numpy as np
import scipy.io as scio


class LegacyMatlabModelMethods:
    """
    遗留的MATLAB模型方法集合
    这些方法原本在DeepEDMD类中，现已移除，仅保留在此文件中
    """
    
    @staticmethod
    def load_parameters(param_path: str):
        """
        加载MATLAB模型参数（.mat/.pkl格式）
        
        注意：此方法已废弃，仅保留用于参考
        """
        if param_path.endswith('.mat'):
            params = scio.loadmat(param_path, squeeze_me=True)
            
            # 优先尝试使用 encoder_weights/decoder_weights (新格式)
            if 'encoder_weights' in params:
                encoder_weights_raw = params['encoder_weights']
                # 处理结构化数组
                if isinstance(encoder_weights_raw, np.ndarray) and encoder_weights_raw.dtype.names:
                    if encoder_weights_raw.shape == ():
                        # 标量结构化数组，转换为字典
                        encoder_dict = {}
                        for name in encoder_weights_raw.dtype.names:
                            value = encoder_weights_raw[name]
                            # 只有当value是0维数组（标量）时才使用item()
                            if isinstance(value, np.ndarray):
                                if value.ndim == 0:
                                    # 0维数组（标量），可以安全使用item()
                                    encoder_dict[name] = value.item()
                                else:
                                    # 多维数组，直接使用，但确保是numpy数组
                                    encoder_dict[name] = np.array(value, dtype=float)
                            elif hasattr(value, 'item'):
                                # 尝试item()，但如果失败则直接使用
                                try:
                                    encoder_dict[name] = value.item()
                                except (ValueError, TypeError):
                                    # item()失败，说明不是标量，直接使用
                                    encoder_dict[name] = np.array(value, dtype=float)
                            else:
                                encoder_dict[name] = value
                        
                        # 将 fc_net.X.weight 和 fc_net.X.bias 映射到 WEF{i+1} 和 bEF{i+1}
                        # 字段名格式：fc_net.0.weight, fc_net.0.bias, fc_net.2.weight, fc_net.2.bias, ...
                        weights = {}
                        biases = {}
                        
                        # 提取所有weight和bias字段，按索引排序
                        weight_keys = [k for k in encoder_dict.keys() if k.endswith('.weight')]
                        bias_keys = [k for k in encoder_dict.keys() if k.endswith('.bias')]
                        
                        # 按照字段中的数字索引排序
                        def extract_index(key):
                            # 从 'fc_net.2.weight' 中提取 2
                            try:
                                parts = key.split('.')
                                return int(parts[1])
                            except:
                                return -1
                        
                        weight_keys_sorted = sorted(weight_keys, key=extract_index)
                        bias_keys_sorted = sorted(bias_keys, key=extract_index)
                        
                        # 映射到 WEF1, WEF2, ... 格式
                        for i, weight_key in enumerate(weight_keys_sorted):
                            weight_value = encoder_dict[weight_key]
                            # 确保权重是numpy数组格式
                            if isinstance(weight_value, np.ndarray):
                                # 如果是0维数组（标量），提取值；否则直接使用
                                if weight_value.ndim == 0:
                                    weight_value = weight_value.item()
                                # 确保是float类型
                                weight_value = np.array(weight_value, dtype=float)
                            else:
                                weight_value = np.array(weight_value, dtype=float)
                            weights[f'WEF{i+1}'] = weight_value
                        
                        for i, bias_key in enumerate(bias_keys_sorted):
                            # 只保存非最后一层的bias（编码器最后一层通常没有bias）
                            if i < len(bias_keys_sorted) - 1:  # 排除最后一层
                                bias_value = encoder_dict[bias_key]
                                # 确保偏置是numpy数组格式
                                if isinstance(bias_value, np.ndarray):
                                    if bias_value.ndim == 0:
                                        bias_value = bias_value.item()
                                    bias_value = np.array(bias_value, dtype=float)
                                else:
                                    bias_value = np.array(bias_value, dtype=float)
                                biases[f'bEF{i+1}'] = bias_value
                    else:
                        # 非标量结构化数组，使用item()方法
                        weights = encoder_weights_raw.item() if hasattr(encoder_weights_raw, 'item') else encoder_weights_raw
                        biases = {}  # 暂时设为空
                elif hasattr(encoder_weights_raw, 'item'):
                    # 如果有item方法，尝试获取
                    item_result = encoder_weights_raw.item()
                    if isinstance(item_result, dict):
                        weights = item_result
                        biases = {}  # 暂时设为空
                    else:
                        weights = item_result
                        biases = {}
                else:
                    weights = encoder_weights_raw
                    biases = {}
                
                # 处理decoder_weights（如果需要）
                if 'decoder_weights' in params:
                    # decoder权重暂时不需要，因为编码器不需要decoder
                    pass
            
            # 回退到旧的 weights/biases 格式
            elif 'weights' in params:
                weights_raw = params['weights']
                # 处理MATLAB结构体
                if hasattr(weights_raw, 'item'):
                    weights = weights_raw.item()
                elif isinstance(weights_raw, np.ndarray) and weights_raw.dtype.names:
                    # 结构化数组，转换为字典
                    if weights_raw.shape == ():
                        # 标量结构化数组
                        weights = {name: weights_raw[name].item() if hasattr(weights_raw[name], 'item') else weights_raw[name] 
                                   for name in weights_raw.dtype.names}
                    else:
                        weights = weights_raw.item() if hasattr(weights_raw, 'item') else weights_raw
                else:
                    weights = weights_raw
                
                if 'biases' in params:
                    biases_raw = params['biases']
                    if hasattr(biases_raw, 'item'):
                        biases = biases_raw.item()
                    elif isinstance(biases_raw, np.ndarray) and biases_raw.dtype.names:
                        if biases_raw.shape == ():
                            biases = {name: biases_raw[name].item() if hasattr(biases_raw[name], 'item') else biases_raw[name] 
                                      for name in biases_raw.dtype.names}
                        else:
                            biases = biases_raw.item() if hasattr(biases_raw, 'item') else biases_raw
                    else:
                        biases = biases_raw
                else:
                    biases = {}
            else:
                # 如果都不存在，设置为空字典
                weights = {}
                biases = {}
            
            # 检查权重结构
            if not isinstance(weights, dict):
                raise ValueError(f"weights类型为 {type(weights)}，不是字典")
            
            # 加载koopman_weights（如果存在）
            if 'koopman_weights' in params:
                koopman_weights = params['koopman_weights']
            else:
                koopman_weights = None
            
            # 网络结构参数
            encoder_widths = params.get('encoder_widths', [])
            decoder_widths = params.get('decoder_widths', [])
            eact_type = params.get('eact_type', [])
            dact_type = params.get('dact_type', [])
            s_dim = int(params.get('s_dim', 3))
            u_dim = int(params.get('u_dim', 2))
            lift_dim = int(params.get('lift_dim', 10))
            conca_num = int(params.get('conca_num', 0))
            state_bound = params.get('state_bound', np.array([[-0.2, -2.7, -1.2], [27.3, 1.9, 1.1]]))
            action_bound = params.get('action_bound', np.array([[-7.9, 0., 0.], [7.9, 0.2, 9.1]]))
            
            return {
                'weights': weights,
                'biases': biases,
                'koopman_weights': koopman_weights,
                'encoder_widths': encoder_widths,
                'decoder_widths': decoder_widths,
                'eact_type': eact_type,
                'dact_type': dact_type,
                's_dim': s_dim,
                'u_dim': u_dim,
                'lift_dim': lift_dim,
                'conca_num': conca_num,
                'state_bound': state_bound,
                'action_bound': action_bound
            }
            
        elif param_path.endswith('.pkl'):
            import pickle
            with open(param_path, 'rb') as f:
                params = pickle.load(f)
            return {
                'weights': params['weights'],
                'biases': params['biases'],
                'encoder_widths': params['encoder_widths'],
                'decoder_widths': params['decoder_widths'],
                'eact_type': params['eact_type'],
                'dact_type': params['dact_type'],
                's_dim': int(params['s_dim']),
                'u_dim': int(params['u_dim']),
                'lift_dim': int(params['lift_dim']),
                'conca_num': int(params['conca_num']),
                'state_bound': params['state_bound'],
                'action_bound': params['action_bound'],
                'koopman_weights': None
            }
        else:
            raise ValueError(f"不支持的文件格式: {param_path}")
    
    @staticmethod
    def build_block_diagonal_A(diag_elements: np.ndarray, space_dim: int) -> np.ndarray:
        """
        将对角线元素构建为2x2块对角矩阵
        
        Args:
            diag_elements: 对角线元素数组，长度为space_dim
            space_dim: 空间维度
            
        Returns:
            A矩阵: (space_dim, space_dim) 的块对角矩阵
            每个2x2块的形式为: [x1, -x2; x2, x1]
        """
        if len(diag_elements) != space_dim:
            raise ValueError(f"对角线元素数量 {len(diag_elements)} 与空间维度 {space_dim} 不匹配")
        
        A = np.zeros((space_dim, space_dim))
        
        # 每2个相邻元素组成一个2x2块
        # 注意：从参数文件读取的对角线元素，每两个组成一个块
        # 块形式: [x1, -x2; x2, x1]
        # 其中 x1 是第一个元素，x2 是第二个元素
        for i in range(0, space_dim, 2):
            if i + 1 < space_dim:
                # 完整的2x2块
                x1 = diag_elements[i]
                x2 = diag_elements[i + 1]
                # 块形式: [x1, -x2; x2, x1]
                A[i, i] = x1
                A[i, i + 1] = -x2
                A[i + 1, i] = x2
                A[i + 1, i + 1] = x1
            else:
                # 最后一个元素单独处理（如果space_dim是奇数）
                A[i, i] = diag_elements[i]
        
        return A
    
    @staticmethod
    def extract_koopman_matrices(koopman_weights, weights, s_dim, lift_dim, u_dim):
        """
        从权重中提取Koopman算子矩阵A和B
        
        注意：此方法已废弃，仅保留用于参考
        """
        # 优先从koopman_weights中提取A和B
        if koopman_weights is not None:
            try:
                A_raw = None
                B_raw = None
                
                # 处理结构化数组
                if isinstance(koopman_weights, np.ndarray):
                    if hasattr(koopman_weights.dtype, 'names') and koopman_weights.dtype.names:
                        # 结构化数组
                        if koopman_weights.shape == ():
                            # 标量结构化数组
                            A_raw = koopman_weights['A'].item()
                            B_raw = koopman_weights['B'].item()
                        else:
                            A_raw = koopman_weights['A']
                            B_raw = koopman_weights['B']
                elif isinstance(koopman_weights, dict):
                    A_raw = koopman_weights.get('A', None)
                    B_raw = koopman_weights.get('B', None)
                
                if A_raw is not None and B_raw is not None:
                    # 转换为numpy数组
                    A_arr = np.array(A_raw, dtype=float)
                    B_arr = np.array(B_raw, dtype=float)
                    
                    # 确保A是矩阵形状
                    space_dim = int(s_dim + lift_dim)  # 确保是整数
                    if A_arr.ndim == 1:
                        # 如果A是一维数组，检查是否需要reshape
                        if A_arr.size == space_dim * space_dim:
                            # 如果是完整的矩阵元素（展平的），reshape为矩阵
                            A_arr = A_arr.reshape(space_dim, space_dim)
                        elif A_arr.size == space_dim:
                            # 如果只有对角线元素，需要转换为2x2块对角结构
                            # MATLAB中A矩阵是块对角矩阵，每2个相邻元素组成一个2x2块
                            # 块形式: [x1, -x2; x2, x1]
                            A_arr = LegacyMatlabModelMethods.build_block_diagonal_A(A_arr, space_dim)
                        else:
                            raise ValueError(f"A矩阵的形状不正确: {A_arr.shape}, 期望 {(space_dim, space_dim)} 或对角线元素 {space_dim}")
                    elif A_arr.ndim == 2:
                        # 如果已经是矩阵，检查是否是块对角结构
                        # 如果不是块对角结构，尝试从对角线元素重建
                        if A_arr.shape == (space_dim, space_dim):
                            # 检查是否是对角矩阵
                            if np.allclose(A_arr, np.diag(np.diag(A_arr))):
                                # 是对角矩阵，需要转换为块对角结构
                                diag_elements = np.diag(A_arr)
                                A_arr = LegacyMatlabModelMethods.build_block_diagonal_A(diag_elements, space_dim)
                            # 否则假设已经是正确的块对角结构
                    
                    # 确保B是正确的形状
                    if B_arr.ndim == 1:
                        if B_arr.size == space_dim * u_dim:
                            B_arr = B_arr.reshape(u_dim, space_dim)
                        else:
                            raise ValueError(f"B矩阵的形状不正确: {B_arr.shape}, 期望 {(u_dim, space_dim)}")
                    
                    # MATLAB代码中使用 ddk.A' 和 ddk.B'，所以这里需要转置
                    A = A_arr.T
                    B = B_arr.T
                    return A, B
            except Exception as e:
                import traceback
                print(f"警告: 从koopman_weights加载A和B矩阵时出错: {e}")
                traceback.print_exc()
        
        # 回退：从权重字典中提取WK和WU
        if isinstance(weights, dict):
            WK = weights.get('WK', None)
            WU = weights.get('WU', None)
        else:
            # MATLAB结构体格式
            WK = weights.WK if hasattr(weights, 'WK') else None
            WU = weights.WU if hasattr(weights, 'WU') else None
        
        if WK is None or WU is None:
            raise ValueError("无法从参数文件中找到WK或WU矩阵（也尝试了koopman_weights中的A和B）")
        
        # 转换为numpy数组
        if not isinstance(WK, np.ndarray):
            WK = np.array(WK)
        if not isinstance(WU, np.ndarray):
            WU = np.array(WU)
        
        # A和B矩阵
        # 检查WK是否是对角矩阵，如果是，需要转换为块对角结构
        space_dim = int(s_dim + lift_dim)
        if WK.shape == (space_dim, space_dim):
            # 检查是否是对角矩阵
            if np.allclose(WK, np.diag(np.diag(WK))):
                # 是对角矩阵，需要转换为块对角结构
                diag_elements = np.diag(WK)
                WK = LegacyMatlabModelMethods.build_block_diagonal_A(diag_elements, space_dim)
        
        A = WK.T  # MATLAB中可能是转置的
        B = WU.T
        
        return A, B
    
    @staticmethod
    def compute_action_bounds(action_bound: np.ndarray):
        """
        计算控制输入的边界
        
        注意：此方法已废弃，仅保留用于参考
        
        Args:
            action_bound: 动作边界数组，形状为 (2, u_dim)
        
        Returns:
            a_max: 最大动作边界，形状为 (1, u_dim)
            a_min: 最小动作边界，形状为 (1, u_dim)
        """
        a_max = action_bound[1, :].reshape(1, -1)
        a_min = action_bound[0, :].reshape(1, -1)
        return a_max, a_min
