import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    实现 Root Mean Square Layer Normalization (RMSNorm)。
    
    RMSNorm(a) = (a / RMS(a)) * weight
    其中 RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        构造 RMSNorm 模块。
        
        参数:
            d_model (int): 模型的隐藏维度。
            eps (float): 用于数值稳定性的 epsilon 值，默认为 1e-5。
            device (torch.device | None): 存储参数的设备。
            dtype (torch.dtype | None): 参数的数据类型。
        """
        super().__init__()
        self.eps = eps
        
        # weight 是一个可学习的 "增益" (gain) 参数，维度为 (d_model,)。
        # 对应于公式中的 g_i。
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行 RMSNorm 归一化。
        
        输入张量 x 的形状预期为 (batch_size, sequence_length, d_model)。
        归一化是沿着最后一个维度（d_model）进行的。
        
        参数:
            x (torch.Tensor): 输入张量。
            
        返回:
            torch.Tensor: 归一化后的张量，形状与 x 相同。
        """
        # 1. 记录原始数据类型并上转到 torch.float32，以防止平方时溢出，这是关键步骤。
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 2. 计算 RMS(a)
        
        # 根据公式：RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
        # PyTorch 的 torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        # 已经计算了平方和的根 (L2 范数)，即 sqrt(sum(a_i^2))。
        #
        # 我们需要先计算平方和：sum(a_i^2)
        # x.pow(2) 计算 a_i^2。
        # .sum(dim=-1, keepdim=True) 沿着 d_model 维度求和，并保持维度以进行广播。
        # d_model 是最后一个维度的大小。
        d_model = x.size(-1)
        
        # 平方和的平均值：1/d_model * sum(a_i^2)
        # 注意：此处使用 mean(dim=-1) 替代了 (sum(a_i^2) / d_model)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 计算 RMS: sqrt(mean_square + eps)
        rms = torch.sqrt(mean_square + self.eps)

        # 3. 计算 RMSNorm(a)
        
        # 根据公式：RMSNorm(a) = (a / RMS(a)) * g
        # (x / rms) 实现了 a_i / RMS(a)。RMS(a) 的形状为 (batch_size, seq_len, 1)，
        # 会通过广播机制自动除以 x 的所有元素。
        normalized_x = x / rms
        
        # 乘以可学习的增益参数 g_i。self.g 的形状是 (d_model,)，
        # 也会通过广播机制自动相乘。
        result = normalized_x * self.weight

        # 4. 下转回原始数据类型并返回结果。
        return result.to(in_dtype)

# 示例使用
if __name__ == '__main__':
    # 定义模型参数
    batch_size = 4
    sequence_length = 10
    d_model = 512
    
    # 假设输入是一个半精度浮点张量 (例如 bfloat16 或 float16)
    # 很多模型会使用这种数据类型来加速训练并减少内存占用
    input_tensor = torch.randn(batch_size, sequence_length, d_model, dtype=torch.bfloat16)

    # 实例化 RMSNorm 模块
    rms_norm_module = RMSNorm(d_model=d_model)

    # 进行前向传播
    output_tensor = rms_norm_module(input_tensor)

    # 打印结果信息
    print("输入张量形状:", input_tensor.shape)
    print("输入张量 dtype:", input_tensor.dtype)
    print("-" * 30)
    print("输出张量形状:", output_tensor.shape)
    print("输出张量 dtype:", output_tensor.dtype)
    print("g 参数形状:", rms_norm_module.weight.shape)
    # 检查输出是否在合理的范围内，RMSNorm 的输出方均根应接近 1
    # 我们可以通过计算输出的 RMS 来验证
    # 注意：这里的检查仅用于示例，实际应用中无需进行
    output_rms = torch.sqrt(output_tensor.to(torch.float32).pow(2).mean(dim=-1)).mean()
    print("输出张量的平均 RMS (应接近 1):", output_rms.item())