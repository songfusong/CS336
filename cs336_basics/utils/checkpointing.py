import torch
import torch.nn as nn
import torch.optim as optim
import typing
import os

def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    iteration: int, 
    out: typing.Union[str, os.PathLike, typing.BinaryIO]
):
    """
    保存模型、优化器状态以及当前迭代次数到指定的文件或文件对象。

    参数:
        model (torch.nn.Module): 要保存的模型实例。
        optimizer (torch.optim.Optimizer): 要保存的优化器实例。
        iteration (int): 当前的训练迭代次数。
        out (str | os.PathLike | typing.BinaryIO): 检查点保存的路径或文件对象。
    """
    # 1. 创建检查点对象（一个包含所有状态信息的字典）
    checkpoint_obj = {
        'model_state_dict': model.state_dict(),     # 模型的学习参数
        'optimizer_state_dict': optimizer.state_dict(), # 优化器的内部状态
        'iteration': iteration,                     # 当前迭代次数
    }
    
    # 2. 使用 torch.save() 将对象保存到指定输出位置
    torch.save(checkpoint_obj, out)
    
    # 可选：打印确认信息
    # if isinstance(out, (str, os.PathLike)):
    #     print(f"检查点成功保存至 {out}，当前迭代次数为 {iteration}")

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO], 
    model: nn.Module, 
    optimizer: optim.Optimizer
) -> int:
    """
    从源文件加载检查点，恢复模型和优化器的状态，并返回保存的迭代次数。

    参数:
        src (str | os.PathLike | typing.BinaryIO): 检查点文件的路径或文件对象。
        model (torch.nn.Module): 要加载状态的目标模型实例。
        optimizer (torch.optim.Optimizer): 要加载状态的目标优化器实例。

    返回:
        int: 检查点中保存的迭代次数。
    """
        
    # 1. 确定模型所在的设备，用于正确加载张量
    # 获取模型第一个参数所在的设备（CPU/CUDA）
    device = next(model.parameters()).device
    
    # 加载检查点对象，并指定 map_location 将张量映射到当前设备
    checkpoint_obj = torch.load(src, map_location=device)
    
    # 2. 恢复模型和优化器状态
    # 使用 .load_state_dict() 方法恢复状态
    model.load_state_dict(checkpoint_obj['model_state_dict'])
    optimizer.load_state_dict(checkpoint_obj['optimizer_state_dict'])
    
    # 3. 恢复迭代次数
    iteration = checkpoint_obj['iteration']

    return iteration