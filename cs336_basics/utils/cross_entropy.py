import torch
from jaxtyping import Float, Int # 假设我们使用 jaxtyping

def cross_entropy(
    logits: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, "batch_size"],
) -> torch.Tensor:
    """
    计算交叉熵损失，并在批次维度上取平均。
    
    Args:
        logits: 预测的 Logits 张量。最后一个维度是词汇表大小 (V)。
                形状为 (B1, B2, ..., V)。
        targets: 目标索引张量。形状为 (B1, B2, ...)。
    
    Returns:
        torch.Tensor: 批次上所有元素的平均交叉熵损失（单个标量）。
    """
    
    # 1. 确保 targets 是 long 类型 (通常是 PyTorch 索引的要求)
    targets_long = targets.long()
    
    # 获取词汇表维度（最后一个维度）
    vocab_dim = -1
    
    # --- 1. 数值稳定性：减去最大元素 M ---
    # 找到 Logits 在词汇表维度上的最大值 M
    # M.shape: (B1, B2, ..., 1)
    M = torch.max(logits, dim=vocab_dim, keepdim=True).values
    
    # Logits 减去 M，确保 exp(logits - M) 中的最大值是 exp(0) = 1
    # shifted_logits.shape: (B1, B2, ..., V)
    shifted_logits = logits - M
    
    # --- 2. 计算 Log-Sum-Exp 的简化形式 ---
    # Log-Sum-Exp (LSE) 的前半部分：log(sum(exp(o_k - M))) + M
    
    # 计算 exp(o_k - M)
    exp_shifted_logits = torch.exp(shifted_logits)
    
    # 求和：sum(exp(o_k - M))
    sum_exp_shifted_logits = torch.sum(exp_shifted_logits, dim=vocab_dim, keepdim=True)
    
    # 求对数：log(sum(exp(o_k - M)))
    log_sum_exp_term = torch.log(sum_exp_shifted_logits)
    
    # LSE 结果 (Log Partition Function): log(sum(exp(o_k)))
    log_partition_function = M + log_sum_exp_term # 形状: (B1, B2, ..., 1)
    
    # --- 3. 提取目标 Logits (o_{i, x_{i+1}}) ---
    # targets.shape: (B1, B2, ...)
    
    # 使用 gather 或类似方法提取目标位置的 Logits
    # 我们需要 log_partition_function 和 logits 具有相同的非词汇表维度形状
    # 使用 torch.gather 或 torch.select/index_select 更易读，但这里直接用索引更简洁
    
    # 提取目标位置的原始 logits 值
    # target_logits.shape: (B1, B2, ...)
    target_logits = torch.gather(logits, dim=vocab_dim, index=targets_long.unsqueeze(vocab_dim)).squeeze(vocab_dim)
    
    # 调整 log_partition_function 形状以匹配 target_logits
    log_partition_function_flat = log_partition_function.squeeze(vocab_dim) # 形状: (B1, B2, ...)
    
    # --- 4. 计算最终损失 (l_i = log(sum(exp(o_k))) - o_{i, x_{i+1}}) ---
    # l_i.shape: (B1, B2, ...)
    loss_elements = log_partition_function_flat - target_logits
    
    # --- 5. 处理批次维度并返回平均值 ---
    # 将所有批次/序列元素展平，然后取平均
    average_loss = torch.mean(loss_elements)
    
    return average_loss

# --- 示例用法 ---
# 假设 batch_size=2, sequence_length=3, vocab_size=5
B, L, V = 2, 3, 5
example_logits = torch.randn(B, L, V)
# 假设目标词汇索引，形状为 (B, L)
example_targets = torch.randint(0, V, (B, L))

loss = cross_entropy(example_logits, example_targets)
# print(f"Average Cross Entropy Loss: {loss.item()}")

# --- 验证 (与 PyTorch 内置函数比较) ---
# PyTorch 内置的 CrossEntropyLoss 期望 Logits 形状为 (N, C) 或 (N, C, ...)
# 并且 Targets 形状为 (N) 或 (N, ...)
loss_fn = torch.nn.CrossEntropyLoss()

# 展平 Logits 和 Targets 以匹配内置函数期望的形状
# N = B * L
logits_flat = example_logits.view(-1, V)
targets_flat = example_targets.view(-1)

# PyTorch 内置损失
torch_loss = loss_fn(logits_flat, targets_flat)
# print(f"PyTorch Built-in Loss: {torch_loss.item()}")
# 您的实现应该非常接近内置实现的值