import torch
import torch.nn as nn
from typing import Optional
from cs336_basics.utils.causalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.utils.positionwise_feedforward import SwiGLUFFN
from cs336_basics.utils.rmsNorm import RMSNorm


# ----------------------------------------------------------------------
# 核心实现：Transformer Block
# ----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    实现预归一化 (pre-norm) Transformer block。
    结构遵循 'RMSNorm -> Operation -> Add'。
    """

    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len: Optional[int] = None, # 传递给 Attention
                 rope_theta: Optional[float] = 10000.0, # 传递给 Attention
                 norm_eps: float = 1e-5):
        """
        初始化 Transformer block。

        参数:
        d_model: int - Transformer block 的输入/输出维度。
        num_heads: int - 多头自注意力中的头数。
        d_ff: int - 位置感知前馈网络 (FFN) 的内层维度。
        max_seq_len: Optional[int] - 传递给 CausalMultiHeadSelfAttention (用于 RoPE)。
        rope_theta: Optional[float] - 传递给 CausalMultiHeadSelfAttention (用于 RoPE)。
        norm_eps: float - RMSNorm 的 epsilon 值。
        """
        super().__init__()
        
        # ------------------------------------
        # 1. Self-Attention 子层
        # ------------------------------------
        self.ln1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            rope_theta=rope_theta
        )
        
        # ------------------------------------
        # 2. Feed-Forward 子层
        # ------------------------------------
        self.ln2 = RMSNorm(d_model, eps=norm_eps)
        # Position-Wise Feed-Forward 使用 SwiGLUFFN 实现
        self.ffn = SwiGLUFFN(
            d_model=d_model, 
            d_ff=d_ff
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        x: torch.Tensor - 输入张量，形状为 (batch_size, seq_len, d_model)。

        返回:
        torch.Tensor - 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        
        # ------------------------------------
        # 1. Self-Attention 子层 (pre-norm 结构)
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        # ------------------------------------
        
        # RMSNorm
        norm_x = self.ln1(x)
        
        # Causal Multi-Head Self-Attention
        attn_out = self.attn(norm_x)
        
        # 残差连接 (Add)
        x = x + attn_out
        
        # ------------------------------------
        # 2. Feed-Forward 子层 (pre-norm 结构)
        # y = x + FFN(RMSNorm(x))
        # ------------------------------------
        
        # RMSNorm
        norm_x_ffn = self.ln2(x)
        
        # Position-Wise Feed-Forward (SwiGLUFFN)
        ffn_out = self.ffn(norm_x_ffn)
        
        # 残差连接 (Add)
        output = x + ffn_out
        
        return output
    

class TransformerBlockTorch(nn.Module):
    """
    使用导入的自定义模块（RMSNorm, CausalMultiHeadSelfAttention, SwiGLUFFN）
    实现的预归一化 (pre-norm) Transformer block。结构遵循 'RMSNorm -> Operation -> Add'。
    接口和逻辑与您原始的 TransformerBlock 完全一致，用于验证封装逻辑。
    """

    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len: Optional[int] = None, # 传递给 Attention
                 rope_theta: Optional[float] = 10000.0, # 传递给 Attention
                 norm_eps: float = 1e-5):
        """
        初始化 Transformer block。参数与原始实现一致。
        """
        super().__init__()
        
        # 1. Self-Attention 子层
        self.ln1 = RMSNorm(d_model, eps=norm_eps) # 使用自定义 RMSNorm
        self.attn = CausalMultiHeadSelfAttention( # 使用自定义 CausalMultiHeadSelfAttention (含 RoPE)
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            rope_theta=rope_theta
        )
        
        # 2. Feed-Forward 子层
        self.ln2 = RMSNorm(d_model, eps=norm_eps) # 使用自定义 RMSNorm
        self.ffn = SwiGLUFFN( # 使用自定义 SwiGLUFFN
            d_model=d_model, 
            d_ff=d_ff
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。逻辑与您原始实现一致。
        """
        
        # ------------------------------------
        # 1. Self-Attention 子层 (pre-norm 结构: x = x + Attn(RMSNorm(x)))
        # ------------------------------------
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x)
        x = x + attn_out # 残差连接
        
        # ------------------------------------
        # 2. Feed-Forward 子层 (pre-norm 结构: x = x + FFN(RMSNorm(x)))
        # ------------------------------------
        norm_x_ffn = self.ln2(x)
        ffn_out = self.ffn(norm_x_ffn)
        output = x + ffn_out # 残差连接
        
        return output

# --- 调试建议 ---
# 如果您用这个类替换掉您的 TransformerBlock，并使用相同的 run_transformer_block 函数运行测试，
# 并且它仍然失败，那么问题出在以下三个地方之一：
# 1. CausalMultiHeadSelfAttention 的实现（包含 RoPE）
# 2. SwiGLUFFN 的实现
# 3. RMSNorm 的实现
# 
# 考虑到您运行了 test_multihead_self_attention_with_rope 和 test_rope 都 PASSED，
# 那么注意力模块（包括RoPE）可能正确。
#
# 最可能的问题仍是：
# 1. SwiGLUFFN 或 RMSNorm 的实现问题。
# 2. 权重加载时的微小差异或快照文件过期。


# 示例使用
if __name__ == '__main__':
    # 定义参数
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 4
    SEQ_LEN = 64
    
    # 实例化 TransformerBlock
    transformer_block = TransformerBlockTorch(
        d_model=D_MODEL, 
        num_heads=NUM_HEADS, 
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # 创建随机输入张量
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    print(f"输入形状: {input_tensor.shape}")
    
    # 前向传播
    output_tensor = transformer_block(input_tensor)
    
    # 打印输出形状
    print(f"输出形状: {output_tensor.shape}")
    
    # 验证输入和输出形状是否一致 (batch_size, seq_len, d_model)
    assert output_tensor.shape == input_tensor.shape
    print("Transformer Block 实现验证成功。")

