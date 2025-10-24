import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from cs336_basics.train_BPE_tokenizer import train_bpe_tokenizer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils.linear import Linear
from cs336_basics.utils.embedding import Embedding
from cs336_basics.utils.rmsNorm import RMSNorm
from cs336_basics.utils.positionwise_feedforward import SwiGLUFFN
from cs336_basics.utils.rotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.utils.softmax import softmax
from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention_einx
from cs336_basics.utils.causalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.utils.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    """
    实现 Transformer Language Model (LM)
    架构: Token Embedding -> num_layers Transformer Blocks -> RMSNorm -> Linear -> Softmax
    """

    def __init__(self,
                 vocab_size: int,
                 context_length: int, # 在这个简化实现中，context_length主要用于Token Embedding (如果需要Positional Embedding)
                                       # 但根据图示和要求，我们主要关注num_layers和TransformerBlock的参数。
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: Optional[float] = 10000.0,
                 device=None,
                 dtype=None):
        """
        初始化 Transformer LM 模块。

        参数:
            vocab_size (int): 词汇表大小。
            context_length (int): 最大上下文长度（用于位置编码，虽然本实现省略了位置编码，但保留接口）。
            num_layers (int): Transformer block 的数量。
            d_model (int): 模型的隐藏维度。
            num_heads (int): Attention 头的数量。
            d_ff (int): 前馈网络中的隐藏维度。
            rope_theta (Optional[float]): 传递给 TransformerBlock 中的 Attention。
            norm_eps (float): 传递给 RMSNorm 和 TransformerBlock 中的 RMSNorm。
        """
        super().__init__()

        # 1. Token Embedding
        # num_embeddings = vocab_size, embedding_dim = d_model
        # 注意：这里的 Embedding 模块被假定为也处理 Positional Embedding (如 Rotary/RoPE)
        # 或者 TransformerBlock 内部处理 RoPE。根据提供的签名，我们只关注 Token Embedding。
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        # 2. Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length, # 传递最大序列长度
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        # 3. Final Normalization (Norm)
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        # 4. Output Linear Layer (Output Embedding)
        # in_features = d_model, out_features = vocab_size
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入的 token ID 序列。形状: (batch_size, sequence_length)

        返回:
            torch.Tensor: 词汇表上的概率分布。形状: (batch_size, sequence_length, vocab_size)
        """
        # 1. Inputs -> Token Embedding
        # x.shape: (B, L) -> h.shape: (B, L, D)
        h = self.token_embeddings(x)

        # 2. Pass through num_layers Transformer Blocks
        # h.shape: (B, L, D) -> h.shape: (B, L, D)
        for block in self.layers:
            h = block(h)

        # 3. Norm
        # h.shape: (B, L, D) -> h.shape: (B, L, D)
        h = self.ln_final(h)

        # 4. Linear (Output Embedding)
        # h.shape: (B, L, D) -> logits.shape: (B, L, V)
        logits = self.lm_head(h)

        return logits

# 注意:
# 1. 实际的 Transformer 语言模型通常输出 logits (在 Softmax 之前)，
# 因为交叉熵损失函数通常直接接收 logits 并在内部执行 log-softmax。
#    然而，**根据提供的架构图**，它明确显示 Softmax 
# 是模型的一部分并输出“Output Probabilities”，因此我们在 forward 中执行 Softmax。
# debug后不执行Softmax
# 2. Positional Encoding（如 RoPE）的实现细节被包含在假设的 
# `Embedding` 或 `TransformerBlock` 内部，因为上下文长度 `context_length` 是必需的输入参数之一。