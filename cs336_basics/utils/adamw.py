import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple, Union
import math

# 定义 AdamW 类，继承自 torch.optim.Optimizer
class AdamW(Optimizer):
    """
    实现了 AdamW 优化器。

    AdamW 是一种随机优化方法，它根据梯度更新参数，并应用解耦的权重衰减。
    它保持了每个参数的运行估计值（动量估计）。
    """

    def __init__(
        self,
        params: Union[Iterable[nn.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        初始化 AdamW 优化器。

        参数:
            params: 待优化的参数（nn.Parameter）或定义参数组的字典的迭代器。
            lr: 学习率（alpha）。
            betas: 计算梯度及其平方的运行平均值的系数（beta1, beta2）。
            eps: 加到分母上的项，用于提高数值稳定性（epsilon）。
            weight_decay: 解耦的权重衰减（lambda）。
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1 参数: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2 参数: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"无效的 epsilon 值: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay 值: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        # 调用父类构造函数，将超参数存储在 self.defaults 中
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        执行单个优化步骤。

        参数:
            closure (可调用对象, 可选): 一个重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 获取组超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历组内所有参数
            for p in group['params']:
                # 跳过没有梯度的参数
                if p.grad is None:
                    continue
                
                # 梯度的别名 g
                grad = p.grad.data
                
                # 访问此参数的状态。self.state 存储了 AdamW 所需的动量估计。
                state = self.state[p]

                # --- 初始化 (init(theta), init(0), init(0)) ---
                if len(state) == 0:
                    # t 从 1 开始
                    state['step'] = 0
                    # 第一动量向量的初始值 m
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 第二动量向量的初始值 v
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # 获取动量估计 m 和 v
                m = state['exp_avg']
                v = state['exp_avg_sq']
                
                # 更新步数计数器 t <- t + 1
                state['step'] += 1
                t = state['step']
                
                # --- 算法 1 循环体步骤 ---
                
                # 1. 更新第一动量估计 (m)
                # m <- beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. 更新第二动量估计 (v)
                # v <- beta2 * v + (1 - beta2) * g^2
                # 使用 addcmul_ 实现 g^2 乘 (1 - beta2) 的累加
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 计算第 t 次迭代的调整后 alpha (alpha_t)
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                
                # 计算偏差校正项
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                # alpha_t
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # 4. 更新参数 (theta)
                # theta <- theta - alpha_t * m / (sqrt(v) + epsilon)
                
                # 分母: sqrt(v) + epsilon
                denom = v.sqrt().add_(eps)
                
                # In-place 参数更新: p <- p - step_size * (m / denom)
                step_size = alpha_t
                p.data.addcdiv_(m, denom, value=-step_size)

                # 5. 应用权重衰减 (lambda)
                # theta <- theta - alpha * lambda * theta
                if weight_decay != 0:
                    # 应用权重衰减（解耦）：使用原始学习率 lr (alpha)
                    # 这是 AdamW 的标准做法
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

# --- 示例用法 (可选，用于测试功能) ---

# if __name__ == '__main__':
#     # 简单的模型和数据
#     p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
#     target = torch.tensor([5.0, 5.0])

#     # 实例化优化器
#     optimizer = AdamW([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

#     print(f"初始参数 p: {p.data}")
    
#     # 简单的训练循环
#     for t in range(1, 6):
#         def closure():
#             # 梯度清零
#             if p.grad is not None:
#                 p.grad.zero_()
            
#             # 简单的 MSE 损失
#             loss = (p - target).pow(2).sum()
#             loss.backward()
#             return loss

#         loss = optimizer.step(closure)
        
#         # 检查状态
#         state = optimizer.state[p]
#         if t == 1:
#             print(f"\n--- 步骤 1 ---")
#             print(f"损失: {loss.item():.4f}")
#             print(f"状态中的步数 t: {state['step']}")
#             print(f"第一动量 (m): {state['exp_avg']}")
#             print(f"第二动量 (v): {state['exp_avg_sq']}")
        
#         if t % 5 == 0:
#             print(f"\n--- 步骤 {t} ---")
#             print(f"参数 p: {p.data}")
#             print(f"损失: {loss.item():.4f}")
            
#     # 检查最终状态
#     print(f"\n最终参数 p: {p.data}")
#     final_state = optimizer.state[p]
#     print(f"最终步数: {final_state['step']}")


def lr_cosine_schedule_with_warmup(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
) -> float:
    """
    实现带 Warm-up 和余弦退火的学习率调度器。

    参数:
        t (int): 当前迭代步数。
        alpha_max (float): 最大学习率 (alpha_max)。
        alpha_min (float): 最小/最终学习率 (alpha_min)。
        T_w (int): Warm-up 阶段的迭代步数 (T_w)。
        T_c (int): Cosine annealing 阶段的总迭代步数 (T_c)。

    返回:
        float: 当前迭代步数 t 对应的学习率 alpha_t。
    """
    
    # 确保 T_c >= T_w，否则退火范围无效
    if T_c < T_w:
        # 在实际应用中，如果 T_c < T_w，通常直接返回 alpha_min 或抛出错误。
        # 这里为了稳健性，直接返回 alpha_min
        print(f"Warning: T_c ({T_c}) < T_w ({T_w}). Returning alpha_min.")
        return alpha_min

    # --- 1. Warm-up 阶段 ---
    # 公式: alpha_t = t / T_w * alpha_max
    if t < T_w:
        # 避免除以零
        if T_w == 0:
            return alpha_max
        return (t / T_w) * alpha_max

    # --- 2. Cosine Annealing (余弦退火) 阶段 ---
    # 公式: alpha_t = alpha_min + 1/2 * (1 + cos(pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    elif T_w <= t <= T_c:
        # Cosine annealing 阶段的有效步数
        T_curr = t - T_w
        # Cosine annealing 阶段的总长度
        T_total = T_c - T_w
        
        # 避免除以零 (如果 T_c = T_w，表示没有退火过程，直接到 Post-annealing)
        if T_total == 0:
            return alpha_min

        # 余弦函数的输入: pi * (t - T_w) / (T_c - T_w)
        cos_input = math.pi * T_curr / T_total
        
        # 衰减因子: 1/2 * (1 + cos(...))
        decay_factor = 0.5 * (1 + math.cos(cos_input))
        
        # 学习率范围: (alpha_max - alpha_min)
        alpha_range = alpha_max - alpha_min
        
        # 最终学习率
        alpha_t = alpha_min + alpha_range * decay_factor
        return alpha_t

    # --- 3. Post-annealing 阶段 ---
    # 公式: alpha_t = alpha_min
    elif t > T_c:
        return alpha_min
        
    return alpha_min # 理论上不会执行到这里，作为安全返回

# --- 示例用法 (测试) ---
# if __name__ == '__main__':
#     # 设定超参数
#     ALPHA_MAX = 1e-3
#     ALPHA_MIN = 1e-5
#     T_W = 100       # Warm-up 100 步
#     T_C = 1000      # 总共 1000 步完成退火
    
#     steps = [0, 50, 100, 101, 550, 1000, 1001]
    
#     print(f"--- Cosine Learning Rate Schedule with Warmup ---")
#     print(f"α_max: {ALPHA_MAX}, α_min: {ALPHA_MIN}, T_W: {T_W}, T_C: {T_C}\n")

#     for t in steps:
#         lr = lr_cosine_schedule_with_warmup(t, ALPHA_MAX, ALPHA_MIN, T_W, T_C)
        
#         if t < T_W:
#             phase = "Warm-up"
#         elif t <= T_C:
#             phase = "Cosine Annealing"
#         else:
#             phase = "Post-annealing"
            
#         print(f"Step t={t:<4} ({phase:<18}): α_t = {lr:.8f}")

def gradient_clipping(parameters: Iterable[nn.Parameter], max_norm: float):
    """
    根据给定的最大范数 M，对所有参数的梯度进行裁剪。

    如果所有参数梯度的 L2 范数 ||g||_2 小于 M，则保留不变；
    否则，将 g 缩小一个因子 M / (||g||_2 + epsilon)。

    参数:
        parameters (Iterable[nn.Parameter]): 包含所有待裁剪梯度的参数列表。
        max_norm (float): 允许梯度的最大 L2 范数 M。
    """
    
    # 定义数值稳定性 epsilon
    epsilon = 1e-6 

    # 1. 收集所有参数的梯度并计算它们的 L2 范数平方和 (squared L2 norm)
    total_norm_sq = 0.0
    grads = []

    # 遍历所有参数，只考虑有梯度的参数
    for p in parameters:
        if p.grad is not None:
            # 展平梯度张量并添加到列表中
            grad = p.grad.data
            grads.append(grad)
            # 累加梯度的 L2 范数平方：||g||^2
            total_norm_sq += grad.pow(2).sum()

    # 如果没有梯度，直接返回
    if total_norm_sq == 0.0:
        return 0.0 # 返回 0.0 表示裁剪前的范数

    # 2. 计算 L2 范数 ||g||_2
    total_norm = torch.sqrt(total_norm_sq)

    # 3. 确定缩放因子 (Scaling Factor)
    
    # 缩放因子计算： M / (||g||_2 + epsilon)
    # 只有当 ||g||_2 > M 时才需要缩放，否则缩放因子应为 1。
    
    # 如果总范数大于最大范数 M
    if total_norm > max_norm:
        # 裁剪因子 = M / (||g||_2 + epsilon)
        clip_factor = max_norm / (total_norm + epsilon)
        
        # 4. 原地修改每个参数的梯度
        # g <- g * clip_factor
        for grad in grads:
            grad.mul_(clip_factor)

    # 返回裁剪前的范数 (通常用于日志记录)
    return total_norm.item()


# --- 示例用法 ---
if __name__ == '__main__':
    # 1. 创建两个模拟参数
    p1 = nn.Parameter(torch.randn(3, requires_grad=True))
    p2 = nn.Parameter(torch.randn(2, requires_grad=True))
    
    # 2. 设置它们的梯度 (模拟反向传播的结果)
    # 故意设置一个较大的梯度来触发裁剪
    p1.grad = torch.ones(3) * 10.0
    p2.grad = torch.ones(2) * 5.0
    
    # 3. 计算未裁剪前的 L2 范数 ||g||_2
    # ||g||_2^2 = (10^2)*3 + (5^2)*2 = 300 + 50 = 350
    # ||g||_2 = sqrt(350) ≈ 18.708
    
    params = [p1, p2]
    MAX_NORM = 5.0 # 设置最大范数 M=5.0
    
    print("--- 裁剪前 ---")
    initial_norm = torch.sqrt(p1.grad.pow(2).sum() + p2.grad.pow(2).sum()).item()
    print(f"初始 L2 范数: {initial_norm:.4f}")
    print(f"p1 梯度: {p1.grad}")
    print(f"p2 梯度: {p2.grad}")
    
    # 4. 执行梯度裁剪
    clipping_norm = gradient_clipping(params, MAX_NORM)
    
    # 5. 计算裁剪后的 L2 范数 ||g'||_2
    final_norm = torch.sqrt(p1.grad.pow(2).sum() + p2.grad.pow(2).sum()).item()
    
    print("\n--- 裁剪后 ---")
    # 裁剪后的范数应该略小于或等于 MAX_NORM
    print(f"裁剪操作返回的初始范数 (用于日志): {clipping_norm:.4f}")
    print(f"最大范数 M: {MAX_NORM:.4f}")
    print(f"最终 L2 范数: {final_norm:.4f}")
    print(f"p1 梯度 (已缩放): {p1.grad}")
    print(f"p2 梯度 (已缩放): {p2.grad}")
    
    assert final_norm <= MAX_NORM + 1e-5, "裁剪后的范数不应超过 MAX_NORM"