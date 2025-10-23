from collections.abc import Callable
from typing import Optional
import torch
import math
from cs336_basics.utils.cross_entropy import cross_entropy

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)  # 初始值为 0
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

def main():
    # 定义可训练参数
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # 实例化自定义优化器
    opt = SGD([weights], lr=1)
    target = torch.ones(weights.shape[0])
    # 训练循环
    for t in range(100):
        opt.zero_grad()  # 清零梯度
        loss = cross_entropy(weights, target)  # 计算损失
        print(loss.cpu().item())  # 打印损失值
        loss.backward()  # 反向传播
        opt.step()  # 更新参数

if __name__ == "__main__":
    main()