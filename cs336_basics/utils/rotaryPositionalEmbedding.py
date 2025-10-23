import torch
from torch import nn
from typing import Optional

class RotaryPositionalEmbedding(nn.Module):
    """
    实现旋转位置嵌入 (RoPE)，如论文 "Roformer: Enhanced Transformer with Rotary
    Position Embedding" (Su et al., 2021) 中所述。
    该模块预先计算正弦和余弦值，并在前向传播过程中将旋转应用于输入张量。
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: Optional[torch.device] = None):
        """
        构建 RoPE 模块，并为正弦和余弦值创建缓冲区。

        Args:
            theta (float): 用于频率几何级数的底数 Θ。
            d_k (int): 查询和键向量的维度。必须是偶数。
            max_seq_len (int): 将要输入的最大序列长度。
            device (torch.device, optional): 存储缓冲区的设备。默认为 None。
        """
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"维度 d_k ({d_k}) 必须是偶数。")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 计算逆频率。公式为 theta^-(2k/d)，其中 k 在 [0, 1, ..., d/2-1] 范围内。
        # 这对应于 theta^-(torch.arange(0, d_k, 2) / d_k)。
        freqs_exponents = torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        freqs = self.theta ** -freqs_exponents

        # 创建一个序列位置的张量 [0, 1, ..., max_seq_len-1]。
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)

        # 计算所有 (位置, 维度) 对的角度。
        # 这会产生一个形状为 (max_seq_len, d_k / 2) 的矩阵。
        angles = torch.outer(t, freqs)

        # 预先计算余弦和正弦值。
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        # 将预计算的值注册为非持久性缓冲区。
        # 它们是模块状态的一部分，但不被视为可学习参数。
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        将 RoPE 应用于输入张量。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)。
                              它可以有任意数量的批处理维度。
            token_positions (torch.Tensor): 一个形状为 (..., seq_len) 的张量，
                                            指定 x 中 token 的绝对位置。

        Returns:
            torch.Tensor: 应用了旋转位置嵌入的张量，
                          与输入 x 的形状相同。
        """
        # 为给定的 token 位置检索 cos 和 sin 值。
        # 索引操作 self.cos_cache[token_positions] 会正确处理
        # 来自 token_positions 的批处理维度。
        # cos 和 sin 的形状将是 (..., seq_len, d_k / 2)。
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # RoPE 的核心思想是旋转成对的特征。
        # 对于一个向量 x = [x_0, x_1, x_2, x_3, ...]，我们旋转 (x_0, x_1), (x_2, x_3) 等特征对。
        # 旋转公式为：
        # x'_2k     = x_2k * cos(theta_k) - x_{2k+1} * sin(theta_k)
        # x'_{2k+1} = x_2k * sin(theta_k) + x_{2k+1} * cos(theta_k)

        # 我们可以通过拆分 x 的最后一个维度来高效地实现这一点。
        # x_even 对应于索引 0, 2, 4, ... 处的特征
        # x_odd  对应于索引 1, 3, 5, ... 处的特征
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 应用旋转。
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # 创建一个与 x 形状相同的空张量来存储结果。
        x_rotated = torch.zeros_like(x)

        # 将旋转后的偶数和奇数部分交错地放回结果张量中。
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated
