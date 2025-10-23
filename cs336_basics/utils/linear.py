import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import trunc_normal_
import math

class Linear(nn.Module):
    """
    实现一个自定义的线性变换模块，遵循 nn.Linear 的接口，但不包含偏置项。
    使用截断正态分布进行权重初始化。
    
    W 的形状为 (in_features, out_features)，以支持输入 X 的右乘 (X @ W)，
    这符合 PyTorch 的行向量约定和内存排序。
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        # 1. 调用父类构造函数
        super().__init__()
        
        # 存储输入和输出维度
        self.in_features = in_features
        self.out_features = out_features
        
        # 定义参数的选项，用于 nn.Parameter 构造
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 2. 构造权重参数 W (不使用 W ⊤)
        # 形状为 (out_features, in_features)，用于 X @ W 乘法
        self.weight = Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        
        # 3. 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        根据指定的规则初始化线性权重：
        N(μ = 0, σ² = 2 / (din + dout))，并截断在 [-3σ, 3σ]。
        """
        d_in = self.in_features
        d_out = self.out_features
        
        # 计算标准差 (σ)
        # 注意: 这里的公式是 Xavier/Glorot 初始化的一种变体 (增加了因子 2)
        std = math.sqrt(2.0 / (d_in + d_out))
        
        # 计算截断范围
        a = -3.0 * std
        b = 3.0 * std
        
        # 使用 torch.nn.init.trunc_normal_ 进行初始化
        # 注意：trunc_normal_ 接受 std (标准差)，而不是方差 (σ²)
        trunc_normal_(self.weight, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用线性变换：y = x @ W
        输入 x 的形状: (..., in_features)
        权重 W 的形状: (in_features, out_features)
        输出 y 的形状: (..., out_features)
        """
        # 3. 使用原生的矩阵乘法运算符 @
        # 由于 W 的形状是 (in, out)，这执行了行向量约定下的线性变换
        return x @ self.weight.T

    def extra_repr(self) -> str:
        """提供自定义模块的简洁表示"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'

# --- 示例使用 ---
if __name__ == '__main__':
    # 示例参数
    IN_F = 512
    OUT_F = 1024
    BATCH_SIZE = 32

    # 实例化自定义线性层
    custom_layer = Linear(IN_F, OUT_F)
    
    print(f"自定义线性层: {custom_layer}")
    print(f"权重 W 形状: {custom_layer.weight.shape}")
    
    # 检查初始化统计量 (用于验证)
    std_expected = math.sqrt(2.0 / (IN_F + OUT_F))
    std_actual = custom_layer.weight.std().item()
    mean_actual = custom_layer.weight.mean().item()
    
    print(f"\n--- 初始化检查 ---")
    print(f"预期标准差 (σ): {std_expected:.6f}")
    print(f"实际权重标准差: {std_actual:.6f}")
    print(f"实际权重均值: {mean_actual:.6f}")
    
    # 模拟输入数据 (Batch, Features)
    input_tensor = torch.randn(BATCH_SIZE, IN_F)
    
    # 前向传播
    output_tensor = custom_layer(input_tensor)
    
    print(f"\n--- 前向传播检查 ---")
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
    
    # 验证输出形状是否正确
    assert output_tensor.shape == (BATCH_SIZE, OUT_F)
    print("输出形状验证成功。")
    torch.nn.init.trunc_normal_