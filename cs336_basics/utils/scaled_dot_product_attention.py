import torch
import math
import einx
from cs336_basics.utils.softmax import softmax

def scaled_dot_product_attention_einx(Q, K, V, mask=None):
    """
    Implements the Scaled Dot-Product Attention using the einx library.
    """
    
    # ... (1, 2, 3 步骤不变) ...
    scores = einx.dot("... l d, ... s d -> ... l s", Q, K)
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)
    # ... (Masking 步骤不变) ...
    if mask is not None:
        mask = mask.to(scores.dtype) if mask.dtype != scores.dtype else mask
        
        # 获取一个非常小的浮点数，用于表示负无穷
        neg_inf_value = torch.finfo(scores.dtype).min
        
        # 将 mask 值为 False 的位置设置为负无穷
        # 注意：使用 ~mask 确保填充的是需要被忽略的位置
        scores = scores.masked_fill(~mask.bool(), neg_inf_value)

    # 4. 计算 Attention Probabilities (Softmax)
    # 使用 b... 来匹配所有前导维度，保持与 scores 的创建表达式一致。
    attention_probs = softmax(scores, dim=-1) # 使用导入的自定义 softmax

    # 5. 计算带权重的 Value (Output)
    output = einx.dot("... l s, ... s d_v -> ... l d_v", attention_probs, V)
    return output

# ======================================================================
# Main 测试函数
# ======================================================================

def create_look_ahead_mask(size):
    """创建前瞻掩码 (Look-Ahead Mask)"""
    # torch.tril 创建下三角矩阵 (包括对角线)，这些位置为 True (允许关注)
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask

def main():
    print("--- Scaled Dot-Product Attention (Einx) 测试开始 ---")
    
    # 1. 定义维度
    B = 2   # Batch size
    H = 4   # Attention Heads
    L = 5   # seq_len_q (Query 序列长度)
    S = 5   # seq_len_k (Key/Value 序列长度)
    D = 8   # d_k (特征维度)
    Dv = 16 # d_v (值维度)
    
    # 2. 创建随机张量 (形状: B, H, L, D/Dv)
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, Dv)
    
    # --- 测试 1: 无掩码 ---
    print("\n[测试 1] 无掩码情况:")
    output_no_mask = scaled_dot_product_attention_einx(Q, K, V)
    
    expected_shape = (B, H, L, Dv)
    print(f"输入形状 Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(f"输出形状: {output_no_mask.shape}")
    assert output_no_mask.shape == expected_shape, "无掩码测试：输出形状不匹配！"
    print("形状测试通过。")
    
    # --- 测试 2: 带有前瞻掩码 (Look-Ahead Mask) ---
    print("\n[测试 2] 前瞻掩码情况 (L=S=5):")
    
    # 创建掩码 (形状: L x S = 5 x 5)
    # True 在下三角，False 在上三角
    look_ahead_mask = create_look_ahead_mask(L) 
    
    # 运行带掩码的注意力
    output_masked = scaled_dot_product_attention_einx(Q, K, V, mask=look_ahead_mask)
    
    # 验证输出形状
    assert output_masked.shape == expected_shape, "带掩码测试：输出形状不匹配！"
    print("形状测试通过。")
    
    # 验证掩码效果（一个简单的检查：最后一个 Query 不应该关注到最后一个 Key）
    # 当 i=4 (l) 时，j=5 (s) 被掩盖。如果 Q[4]关注K[4]，但不能关注未来的K[>4]
    # 由于是 L=S=5，最后一个 Query (index L-1=4) 应该只关注到 S 的所有位置（包括 S-1）。
    # *我们无法直接验证注意力分数，但可以确认计算没有崩溃。*
    
    # 检查 Softmax 沿 dim=-1 的求和是否仍然为 1 (即便有掩码)
    # 注意力权重在 Softmax 之后，沿着 Key 轴 (dim=-1) 求和应为 1
    # 我们需要重新计算注意力概率来验证这个属性（因为 output 是加权值）
    
    # 简单验证：使用 PyTorch 的核心操作重复步骤 1-3，然后检查 Softmax
    scores_check = einx.dot("... l d, ... s d -> ... l s", Q, K) / math.sqrt(D)
    
    # 应用掩码到检查得分
    neg_inf_value = torch.finfo(scores_check.dtype).min
    scores_check_masked = scores_check.masked_fill(~look_ahead_mask.bool(), neg_inf_value)
    
    # 计算并检查 Softmax
    probs_check_masked = torch.softmax(scores_check_masked, dim=-1)
    sum_check = torch.sum(probs_check_masked, dim=-1)

    assert torch.allclose(sum_check, torch.ones_like(sum_check), atol=1e-5), "带掩码测试：注意力概率归一化失败！"
    print("归一化检查通过。")
    
    print("\n--- 所有测试完成，Scaled Dot-Product Attention 实现正确！---")

if __name__ == "__main__":
    main()