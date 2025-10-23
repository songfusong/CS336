# import torch
# import torch.nn as nn
# import math
# import einx
# from typing import Optional
# from cs336_basics.utils.rotaryPositionalEmbedding import RotaryPositionalEmbedding
# from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention_einx, create_look_ahead_mask
# from cs336_basics.utils.linear import Linear

# class CausalMultiHeadSelfAttention(nn.Module):
#     """
#     实现因果多头自注意力 (Causal Multi-Head Self-Attention)
#     - 可选地使用 Rotary Positional Embedding (RoPE)。
#     - 实现因果 (Look-Ahead) 掩码。
#     """
#     def __init__(self, d_model: int, num_heads: int, 
#                  max_seq_len: Optional[int] = None, # 默认不使用 RoPE
#                  rope_theta: Optional[float] = 10000.0): # 默认不使用 RoPE
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_head = d_model // num_heads  # d_k = d_v = d_model / num_heads

#         # 线性层 Wq, Wk, Wv 的合并表示。
#         self.q_proj, self.k_proj, self.v_proj = Linear(d_model, d_model), Linear(d_model, d_model), Linear(d_model, d_model)
#         #得到self.qkv
#         d_qkv = 3 * d_model
#         self.qkv_proj = Linear(d_model, d_qkv)
#         new_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
#         self.qkv_proj.weight.data.copy_(new_weight)

#         # 2. 输出投影
#         self.output_proj = Linear(d_model, d_model)
        
#         # 旋转位置嵌入 RoPE 的配置
#         self.use_rope = False
#         self.rope = None
        
#         # 只有在指定 max_seq_len 时才初始化 RoPE
#         if max_seq_len is not None:
#             if max_seq_len <= 0:
#                  raise ValueError("max_seq_len must be positive if provided.")
#             if rope_theta is None:
#                  rope_theta = 10000.0 # 默认值
                 
#             self.use_rope = True
#             self.rope = RotaryPositionalEmbedding(
#                 theta=rope_theta, 
#                 d_k=self.d_head, 
#                 max_seq_len=max_seq_len
#             )
#             print(f"初始化 CausalMultiHeadSelfAttention 时启用了 RoPE (max_seq_len={max_seq_len}, theta={rope_theta})")
#         else:
#             print("初始化 CausalMultiHeadSelfAttention 时未启用 RoPE。")
        
#         # 注册因果掩码 (下三角矩阵，True 允许关注)。大小取一个合理的最大值，如 2048。
#         # 注意：无论是否使用 RoPE，因果注意力都需要因果掩码。
#         buffer_max_len = max_seq_len if max_seq_len is not None else 2048
#         causal_mask = create_look_ahead_mask(buffer_max_len)
#         self.register_buffer('causal_mask', causal_mask, persistent=False)


#     def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         x: 输入张量, 形状 (batch_size, sequence_length, d_model) -> "b t d_model"
#         padding_mask: 可选的额外填充掩码。形状 (b, t)。
#                       True 表示有效 token，False 表示 padding。
#         """
#         B, T, D = x.shape
#         H = self.num_heads
#         D_head = self.d_head
        
#         # 1. QKV 投影和拆分
#         # qkv: "b t d_model -> b t (3 d_model)"
#         qkv = self.qkv_proj(x)
        
#         # 拆分 QKV，并重排维度到 (B, H, T, D_head)
#         # q, k, v 形状: "b h t d_head"
#         q, k, v = einx.rearrange("b t (three h d_head) -> three b h t d_head", qkv, three=3, h=H, d_head=D_head)

#         # 2. 应用 RoPE (可选)
#         if self.use_rope:
#             # **修正点：确保 token_positions 具有批次维度 (B, T)**
            
#             # 1. 生成基础的位置索引 (T,)
#             # 使用 dtype=torch.long 确保是整数索引
#             token_positions_base = torch.arange(T, device=x.device, dtype=torch.long)
            
#             # 2. 扩展到批次维度 (B, T)
#             # unsqueeze(0) 得到 (1, T)，然后 expand(B, T) 得到 (B, T)
#             token_positions = token_positions_base.unsqueeze(0).expand(B, T)
            
#             # 将 RoPE 应用于 Q 和 K。Q/K 形状: (B, H, T, D_head)
#             q_final = self.rope(q, token_positions)
#             k_final = self.rope(k, token_positions)
#         else:
#             # 不使用 RoPE，直接使用原始的 Q 和 K
#             q_final = q
#             k_final = k

#         # 3. 创建最终掩码
#         # 从预注册的缓冲区中切片出当前序列长度的因果掩码 (T, T)
#         causal_mask_sliced = self.causal_mask[:T, :T] 
        
#         # 结合因果掩码和填充掩码
#         if padding_mask is not None:
#              # Key 的 Padding Mask (应用于 scores 的最后一个轴): (B, 1, 1, T)
#              padding_mask_k = padding_mask.unsqueeze(1).unsqueeze(2) 
             
#              # 最终的 attention_mask: (B, 1, T, T)。True 表示允许关注。
#              # 注意：因果掩码是 (T, T)，需要扩展到 (1, 1, T, T) 才能与 padding_mask_k (B, 1, 1, T) 正确广播。
#              attention_mask = padding_mask_k & causal_mask_sliced.unsqueeze(0).unsqueeze(0)
#         else:
#             # 只有因果掩码
#             attention_mask = causal_mask_sliced 

#         # 4. 计算注意力
#         # Q_final, K_final, V 形状: (B, H, T, D_head)
#         # mask 形状: (T, T) 或 (B, 1, T, T)
#         attn_output = scaled_dot_product_attention_einx(q_final, k_final, v, mask=attention_mask)
        
#         # 5. 拼接和最终投影
#         # 合并头维度: "b h t d_head -> b t (h d_head)"
#         concat_output = einx.rearrange("b h t d_head -> b t (h d_head)", attn_output)

#         # 最终投影: "b t d_model -> b t d_model"
#         output = self.output_proj(concat_output)
        
#         return output

# # ======================================================================
# # 运行示例
# # ======================================================================

# def run_example_rope():
#     d_model = 512
#     num_heads = 8
#     max_seq_len = 512 # 提供 max_seq_len，启用 RoPE
#     batch_size = 4
#     seq_len = 10 

#     print("--- 示例 1: 启用 RoPE ---")
#     # 实例化模块 (启用 RoPE)
#     mha_rope = CausalMultiHeadSelfAttention(
#         d_model=d_model, 
#         num_heads=num_heads, 
#         max_seq_len=max_seq_len, # 传入 max_seq_len
#         rope_theta=10000.0
#     )
#     print(f"mha_rope.use_rope: {mha_rope.use_rope}")

#     input_tensor = torch.randn(batch_size, seq_len, d_model)
#     output_tensor = mha_rope(input_tensor)
#     print(f"启用 RoPE 输出形状: {output_tensor.shape}")
#     assert output_tensor.shape == (batch_size, seq_len, d_model)
#     print("形状验证通过。\n")

# def run_example_no_rope():
#     d_model = 512
#     num_heads = 8
#     batch_size = 4
#     seq_len = 10 

#     print("--- 示例 2: 不使用 RoPE (默认) ---")
#     # 实例化模块 (不传入 max_seq_len，默认不使用 RoPE)
#     mha_no_rope = CausalMultiHeadSelfAttention(
#         d_model=d_model, 
#         num_heads=num_heads
#         # max_seq_len 默认为 None
#     )
#     print(f"mha_no_rope.use_rope: {mha_no_rope.use_rope}")
    
#     input_tensor = torch.randn(batch_size, seq_len, d_model)
#     output_tensor = mha_no_rope(input_tensor)
#     print(f"不使用 RoPE 输出形状: {output_tensor.shape}")
#     assert output_tensor.shape == (batch_size, seq_len, d_model)
#     print("形状验证通过。")


# if __name__ == "__main__":
#     run_example_rope()
#     run_example_no_rope()

import torch
import torch.nn as nn
import math
import einx
from typing import Optional
from cs336_basics.utils.rotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention_einx, create_look_ahead_mask
from cs336_basics.utils.linear import Linear

class CausalMultiHeadSelfAttention(nn.Module):
    """
    实现因果多头自注意力 (Causal Multi-Head Self-Attention)
    - 可选地使用 Rotary Positional Embedding (RoPE)。
    - 实现因果 (Look-Ahead) 掩码。
    """
    def __init__(self, d_model: int, num_heads: int, 
                 max_seq_len: Optional[int] = None, # 默认不使用 RoPE
                 rope_theta: Optional[float] = 10000.0): # 默认不使用 RoPE
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / num_heads

        # 线性层 Wq, Wk, Wv 的合并表示。
        self.q_proj, self.k_proj, self.v_proj = Linear(d_model, d_model), Linear(d_model, d_model), Linear(d_model, d_model)
        #得到self.qkv
        # d_qkv = 3 * d_model
        # self.qkv_proj = Linear(d_model, d_qkv)
        # new_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        # self.qkv_proj.weight.data.copy_(new_weight)

        # 2. 输出投影
        self.output_proj = Linear(d_model, d_model)
        
        # 旋转位置嵌入 RoPE 的配置
        self.use_rope = False
        self.rope = None
        
        # 只有在指定 max_seq_len 时才初始化 RoPE
        if max_seq_len is not None:
            if max_seq_len <= 0:
                 raise ValueError("max_seq_len must be positive if provided.")
            if rope_theta is None:
                 rope_theta = 10000.0 # 默认值
                 
            self.use_rope = True
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, 
                d_k=self.d_head, 
                max_seq_len=max_seq_len
            )
            print(f"初始化 CausalMultiHeadSelfAttention 时启用了 RoPE (max_seq_len={max_seq_len}, theta={rope_theta})")
        else:
            print("初始化 CausalMultiHeadSelfAttention 时未启用 RoPE。")
        
        # 注册因果掩码 (下三角矩阵，True 允许关注)。大小取一个合理的最大值，如 2048。
        # 注意：无论是否使用 RoPE，因果注意力都需要因果掩码。
        buffer_max_len = max_seq_len if max_seq_len is not None else 2048
        causal_mask = create_look_ahead_mask(buffer_max_len)
        self.register_buffer('causal_mask', causal_mask, persistent=False)


    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: 输入张量, 形状 (batch_size, sequence_length, d_model) -> "b t d_model"
        padding_mask: 可选的额外填充掩码。形状 (b, t)。
                      True 表示有效 token，False 表示 padding。
        """
        B, T, D = x.shape
        H = self.num_heads
        D_head = self.d_head
        
        # 1. QKV 投影和拆分
        # qkv: "b t d_model -> b t (3 d_model)"
        # qkv = self.qkv_proj(x)
        q = self.q_proj(x) # (B, T, D)
        k = self.k_proj(x) # (B, T, D)
        v = self.v_proj(x) # (B, T, D)
        # 拆分 QKV，并重排维度到 (B, H, T, D_head)
        # q, k, v 形状: "b h t d_head"
        # q, k, v = einx.rearrange("b t (three h d_head) -> three b h t d_head", qkv, three=3, h=H, d_head=D_head)
        # 2. 维度重排：将 Q, K, V 拆分为多头形式 (B, H, T, D_head)
        # 注意：这里的 'd_model' 实际是 (H * D_head)
        q = einx.rearrange("b t (h d_head) -> b h t d_head", q, h=H, d_head=D_head)
        k = einx.rearrange("b t (h d_head) -> b h t d_head", k, h=H, d_head=D_head)
        v = einx.rearrange("b t (h d_head) -> b h t d_head", v, h=H, d_head=D_head)


        # 2. 应用 RoPE (可选)
        if self.use_rope:
            # **修正点：确保 token_positions 具有批次维度 (B, T)**
            
            # 1. 生成基础的位置索引 (T,)
            # 使用 dtype=torch.long 确保是整数索引
            token_positions_base = torch.arange(T, device=x.device, dtype=torch.long)
            
            # 2. 扩展到批次维度 (B, T)
            # unsqueeze(0) 得到 (1, T)，然后 expand(B, T) 得到 (B, T)
            token_positions = token_positions_base.unsqueeze(0).expand(B, T)
            
            # 将 RoPE 应用于 Q 和 K。Q/K 形状: (B, H, T, D_head)
            q_final = self.rope(q, token_positions)
            k_final = self.rope(k, token_positions)
        else:
            # 不使用 RoPE，直接使用原始的 Q 和 K
            q_final = q
            k_final = k

        # 3. 创建最终掩码
        # 从预注册的缓冲区中切片出当前序列长度的因果掩码 (T, T)
        causal_mask_sliced = self.causal_mask[:T, :T] 
        
        # 结合因果掩码和填充掩码
        if padding_mask is not None:
             # Key 的 Padding Mask (应用于 scores 的最后一个轴): (B, 1, 1, T)
             padding_mask_k = padding_mask.unsqueeze(1).unsqueeze(2) 
             
             # 最终的 attention_mask: (B, 1, T, T)。True 表示允许关注。
             # 注意：因果掩码是 (T, T)，需要扩展到 (1, 1, T, T) 才能与 padding_mask_k (B, 1, 1, T) 正确广播。
             attention_mask = padding_mask_k & causal_mask_sliced.unsqueeze(0).unsqueeze(0)
        else:
            # 只有因果掩码
            attention_mask = causal_mask_sliced 

        # 4. 计算注意力
        # Q_final, K_final, V 形状: (B, H, T, D_head)
        # mask 形状: (T, T) 或 (B, 1, T, T)
        attn_output = scaled_dot_product_attention_einx(q_final, k_final, v, mask=attention_mask)
        
        # 5. 拼接和最终投影
        # 合并头维度: "b h t d_head -> b t (h d_head)"
        concat_output = einx.rearrange("b h t d_head -> b t (h d_head)", attn_output)

        # 最终投影: "b t d_model -> b t d_model"
        output = self.output_proj(concat_output)
        
        return output

# ======================================================================
# 运行示例
# ======================================================================

def run_example_rope():
    d_model = 512
    num_heads = 8
    max_seq_len = 512 # 提供 max_seq_len，启用 RoPE
    batch_size = 4
    seq_len = 10 

    print("--- 示例 1: 启用 RoPE ---")
    # 实例化模块 (启用 RoPE)
    mha_rope = CausalMultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        max_seq_len=max_seq_len, # 传入 max_seq_len
        rope_theta=10000.0
    )
    print(f"mha_rope.use_rope: {mha_rope.use_rope}")

    input_tensor = torch.randn(batch_size, seq_len, d_model)
    output_tensor = mha_rope(input_tensor)
    print(f"启用 RoPE 输出形状: {output_tensor.shape}")
    assert output_tensor.shape == (batch_size, seq_len, d_model)
    print("形状验证通过。\n")

def run_example_no_rope():
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 10 

    print("--- 示例 2: 不使用 RoPE (默认) ---")
    # 实例化模块 (不传入 max_seq_len，默认不使用 RoPE)
    mha_no_rope = CausalMultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads
        # max_seq_len 默认为 None
    )
    print(f"mha_no_rope.use_rope: {mha_no_rope.use_rope}")
    
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    output_tensor = mha_no_rope(input_tensor)
    print(f"不使用 RoPE 输出形状: {output_tensor.shape}")
    assert output_tensor.shape == (batch_size, seq_len, d_model)
    print("形状验证通过。")


if __name__ == "__main__":
    run_example_rope()
    run_example_no_rope()
