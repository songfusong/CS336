import torch
from torch import nn
import torch.nn.functional as F
from cs336_basics.utils.linear import Linear

class SwiGLUFFN(nn.Module):
    """
    使用 nn.Parameter 直接定义权重 W1, W3, W2 的 SwiGLU FFN 实现。
    
    FFN(x) = W_2 (SiLU(x W_1) * (x W_3))
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        # 修正 1：将 d_model 和 d_ff 保存为实例属性
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 用于 nn.Parameter 初始化的参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # ----------------------------------------------------
        # 定义权重参数 W1, W3, W2
        # 形状遵循 F.linear(input, weight) 的惯例: (out_features, in_features)
        # ----------------------------------------------------
        
        # W1: d_model -> d_ff。权重形状应为 (d_ff, d_model)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        
        # W3: d_model -> d_ff
        self.w3 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        
        # W2: d_ff -> d_model
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行 SwiGLU FFN 变换。
        """
        # 使用 nn.Linear 实例进行前向传播
        # layer(input) 内部执行的就是 input @ layer.weight.T + layer.bias
        
        # 分支 A: x W_1 (用于 SiLU 激活)
        a = self.w1(x) # 等价于 F.linear(x, self.w1.weight, None)
        
        # 分支 G: x W_3 (用于门控)
        g = self.w3(x) # 等价于 F.linear(x, self.w3.weight, None)
        
        # 2. 实现 SiLU(u) = u * sigmoid(u)
        silu_a = a * torch.sigmoid(a)
        
        # 3. 元素级乘法 (形状: (b, t, d_ff))
        glu_output = silu_a * g
        
        # 4. 最终收缩
        result = self.w2(glu_output) # 等价于 F.linear(glu_output, self.w2.weight, None)
        
        return result

# 示例使用 (如果这是脚本的主体部分)
if __name__ == '__main__':
    # 定义参数
    d_model = 512
    # d_ff 约为 8/3 * d_model，并是 64 的倍数
    d_ff = 1344 
    
    batch_size = 4
    sequence_length = 100
    
    # 创建随机输入张量
    input_tensor = torch.randn(batch_size, sequence_length, d_model)

    # 实例化 SwiGLU FFN 模块
    swiglu_ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff) # 错误 1: 发生在这里，但现在已修复

    # 进行前向传播
    output_tensor = swiglu_ffn(input_tensor) # 错误 2: 发生在 forward 中，现在已修复

    # 打印结果信息
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)
    assert input_tensor.shape == output_tensor.shape