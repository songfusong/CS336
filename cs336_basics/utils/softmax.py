import torch

def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    实现具有数值稳定性的 softmax 操作。
    
    Args:
        tensor (torch.Tensor): 输入张量。
        dim (int): 沿其应用 softmax 的维度索引。
        
    Returns:
        torch.Tensor: 应用了 softmax 的张量，其形状与输入张量相同。
    """
    # 1. 查找最大值 (Trick for Numerical Stability)
    # 查找指定维度上的最大值。
    # keepdim=True 保持了被操作维度的大小为 1，
    # 这样 max_val 的形状就可以与输入张量 tensor 进行广播 (broadcasting)。
    max_val = torch.max(tensor, dim=dim, keepdim=True).values

    # 2. 减去最大值 (The "Trick")
    # 这一步是实现数值稳定性的关键。
    # e^(v_i) 变为 e^(v_i - max(v))，这确保了指数运算的输入最大为 0，
    # 从而避免了 e^(v_i) 变得太大导致溢出 (inf)。
    # 减去 max_val 后，新的最大值将是 0，e^0 = 1。
    exps = torch.exp(tensor - max_val)

    # 3. 计算分母 (Normalization Term)
    # 对经过指数化的结果沿着同一维度求和。
    # keepdim=True 保持形状可用于广播。
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)

    # 4. 计算 softmax
    # 将指数化的值除以它们的和。
    softmax_output = exps / sum_exps

    return softmax_output

# ======================================================================
# 示例和验证
# ======================================================================

def main():
    # 创建一个用于测试的随机张量
    # 假设 batch_size=2, seq_len=4, d_model=3
    test_tensor = torch.randn(2, 4, 3) 
    print(f"原始张量形状: {test_tensor.shape}")

    # 1. 沿着最后一个维度 (dim=2) 应用 softmax
    dim_to_softmax = 2
    custom_softmax = softmax(test_tensor, dim=dim_to_softmax)
    torch_softmax = torch.softmax(test_tensor, dim=dim_to_softmax)

    # 验证形状是否相同
    print(f"自定义 softmax 形状: {custom_softmax.shape}")
    print(f"PyTorch softmax 形状: {torch_softmax.shape}")
    print(f"形状是否匹配: {custom_softmax.shape == torch_softmax.shape}")

    # 验证结果是否在数值上匹配 PyTorch 的内置函数
    # 使用 torch.allclose 来检查数值上的微小差异（浮点精度）
    is_close = torch.allclose(custom_softmax, torch_softmax, atol=1e-6)
    print(f"自定义结果与 PyTorch 内置函数是否匹配 (atol=1e-6): {is_close}")

    # 验证结果是否归一化 (沿 dim=2 求和应该接近 1.0)
    sum_check = torch.sum(custom_softmax, dim=dim_to_softmax)
    print(f"归一化检查 (沿 dim={dim_to_softmax} 求和):")
    print(sum_check)
    print(f"是否所有和都接近 1.0: {torch.allclose(sum_check, torch.ones_like(sum_check))}")

if __name__ == "__main__":
    main()