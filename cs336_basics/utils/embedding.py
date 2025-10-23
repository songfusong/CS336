import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import trunc_normal_
import math

class Embedding(nn.Module):
    """
    实现一个自定义的线性变换模块，遵循 nn.Linear 的接口，但不包含偏置项。
    使用截断正态分布进行权重初始化。
    
    W 的形状为 (in_features, out_features)，以支持输入 X 的右乘 (X @ W)，
    这符合 PyTorch 的行向量约定和内存排序。
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # 1. 调用父类构造函数
        super().__init__()
        
        # 存储输入和输出维度
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 定义参数的选项，用于 nn.Parameter 构造
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 2. 构造embedding matrix
        self.weight = Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        )
        
        # 3. 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        根据指定的规则初始化线性权重：
        N(μ = 0, σ² = 2 / (din + dout))，并截断在 [-3σ, 3σ]。
        """
        d_in = self.num_embeddings
        d_out = self.embedding_dim
        
        # 计算标准差 (σ)
        # 注意: 这里的公式是 Xavier/Glorot 初始化的一种变体 (增加了因子 2)
        std = math.sqrt(2.0 / (d_in + d_out))
        
        # 计算截断范围
        a = -3.0 * std
        b = 3.0 * std
        
        # 使用 torch.nn.init.trunc_normal_ 进行初始化
        # 注意：trunc_normal_ 接受 std (标准差)，而不是方差 (σ²)
        trunc_normal_(self.weight, mean=0.0, std=std, a=a, b=b)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        The forward method should select the embedding vector for 
        each token ID by indexing into an embedding matrix of 
        shape (vocab_size, d_model) using a torch.LongTensor 
        of token IDs with shape (batch_size, sequence_length).
        """
        # 【修正点 3】：简化 token_ids 转换。确保它是 LongTensor 用于索引。
        if token_ids.dtype != torch.long:
             token_ids = token_ids.long()

        return self.weight[token_ids]
    
# --- 示例用法 (Example Usage) ---
if __name__ == '__main__':
    # 定义参数
    vocab_size = 10000
    d_model = 512
    batch_size = 4
    seq_len = 20

    # 实例化自定义 Embedding 模块
    custom_emb = Embedding(vocab_size, d_model)
    print(f"Custom Embedding 矩阵形状: {custom_emb.weight.shape}")

    # 构造模拟输入 (0 到 vocab_size-1 之间的 token ID)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入 Token IDs 形状: {input_ids.shape}, dtype: {input_ids.dtype}")

    # 前向传播
    output_embeddings = custom_emb(input_ids)

    # 检查输出形状
    print(f"输出嵌入向量形状: {output_embeddings.shape}")
    
    # 验证形状是否符合要求 (B, L, D)
    assert output_embeddings.shape == (batch_size, seq_len, d_model)
    print("形状验证通过。")
    
    # 对比 nn.Embedding 的形状
    nn_emb = nn.Embedding(vocab_size, d_model)
    nn_output = nn_emb(input_ids)
    assert nn_output.shape == output_embeddings.shape
    print(f"与 nn.Embedding 输出形状一致: {nn_output.shape}")

