import numpy as np
import torch
# 导入 NumPy 用于数据操作和 PyTorch 用于张量处理

def get_batch(
    x: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一个长的 token ID 序列中，随机采样一个批次的输入序列和对应的下一个 token 目标序列。

    Args:
        x (np.ndarray): 完整的 1D token ID 数组（整数数组）。
        batch_size (int): 批次中的序列数量 (B)。
        context_length (int): 每个序列的长度 (m)。
        device (str): PyTorch 设备字符串 ('cpu', 'cuda:0' 等)。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 一对张量：(input_sequences, target_sequences)。
            两个张量形状都为 (batch_size, context_length)。
    """
    
    # 1. 确定有效起始索引的最大值
    # 我们需要 'context_length' 的输入 token 和 'context_length' 的目标 token。
    # 目标序列 Y 的最后一个 token 位于原始序列 x 的索引 (起始索引 + context_length)。
    # 确保这个索引不会超出 x 的边界 (len(x) - 1)。
    # 因此，输入序列的起始索引 max_start_idx 最大不能超过 len(x) - context_length - 1
    max_start_idx = len(x) - context_length - 1
    
    if max_start_idx < 0:
        raise ValueError(
            f"Input sequence length ({len(x)}) is too short for the requested "
            f"context length ({context_length}). Need at least {context_length + 1} tokens."
        )

    # 2. 随机采样 'batch_size' 个起始索引
    # 这些索引将是输入序列 (X) 的起始位置。
    start_indices = np.random.randint(
        low=0, high=max_start_idx + 1, size=batch_size
    )

    # 初始化 NumPy 数组来存放批次数据
    input_batch_np = np.empty((batch_size, context_length), dtype=x.dtype)
    target_batch_np = np.empty((batch_size, context_length), dtype=x.dtype)

    # 3. 循环提取批次中的每个序列
    for i, start_idx in enumerate(start_indices):
        # 输入序列 (X): [x_i, x_{i+1}, ..., x_{i+m-1}]
        input_batch_np[i, :] = x[start_idx : start_idx + context_length]

        # 目标序列 (Y): [x_{i+1}, x_{i+2}, ..., x_{i+m}]
        # 目标序列是输入序列整体向右错位一个 token 的结果。
        target_batch_np[i, :] = x[start_idx + 1 : start_idx + 1 + context_length]

    # 4. 将 NumPy 数组转换为 PyTorch 张量并放置到指定设备
    # 转换为 PyTorch 张量
    input_tensor = torch.from_numpy(input_batch_np).to(device)
    target_tensor = torch.from_numpy(target_batch_np).to(device)
    
    # 确保张量是整数类型 (如 torch.long)，这是 token ID 常用类型
    # 使用 .long() 确保数据类型为 64 位整数，适合索引或 token ID
    if not input_tensor.is_floating_point():
        input_tensor = input_tensor.long()
    if not target_tensor.is_floating_point():
        target_tensor = target_tensor.long()

    return input_tensor, target_tensor