import torch
import torch.nn as nn
import os
import time
import pickle
import numpy as np
import math
from tqdm import tqdm
from typing import Optional, List, Dict, Any

# --- 导入您提供的所有模块 ---
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
# 假设 TransformerLM 所在文件定义了 TransformerConfig
try:
    from cs336_basics.transformer_lm import TransformerConfig
except ImportError:
    # 临时定义，如果 TransformerLM 文件没有导出它
    class TransformerConfig:
        def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, dropout):
            self.vocab_size = vocab_size
            self.context_length = context_length
            self.num_layers = num_layers
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.dropout = dropout
        def to_dict(self): return self.__dict__
            
from cs336_basics.utils.cross_entropy import cross_entropy
from cs336_basics.utils.adamw import AdamW, lr_cosine_schedule_with_warmup, gradient_clipping
from cs336_basics.utils.data_loading import get_batch
from cs336_basics.utils.checkpointing import load_checkpoint, save_checkpoint


# --- 训练配置常量 ---
MAX_STEPS = 50000       
WARMUP_STEPS = 1000     
EVAL_INTERVAL = 1000    
LOG_INTERVAL = 10       
CHECKPOINT_INTERVAL = 5000 
GRAD_CLIP_NORM = 1.0     

BATCH_SIZE = 64
BLOCK_SIZE = 128         
LEARNING_RATE = 6e-4     
WEIGHT_DECAY = 0.1
DATA_SPLIT_RATIO = 0.9   

# --- 文件路径 ---
VOCAB_FILE = 'cs336_basics/TinyStoriesTrainToken_vocab.pkl'
MERGES_FILE = 'cs336_basics/TinyStoriesTrainToken_merges.pkl'
RAW_DATA_PATH = 'data/TinyStoriesV2-GPT4-train.txt' 
TOKENIZED_DATA_PATH = 'train/tinystories_train_tokens.npy' # 我们将使用 uint16 dtype

# --- 模型配置 (请根据实际作业要求调整) ---
MODEL_CONFIG = TransformerConfig(
    vocab_size=None,     
    context_length=BLOCK_SIZE,
    num_layers=6,
    d_model=384,
    num_heads=6,
    d_ff=384 * 4,        
    dropout=0.0
)

# --- 辅助函数：加载分词器 ---

def load_tokenizer_and_update_config(config: TransformerConfig):
    """从 .pkl 文件加载 BPE 分词器并更新模型配置中的词汇表大小。"""
    
    if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
        raise FileNotFoundError(f"未找到 BPE 文件：{VOCAB_FILE} 或 {MERGES_FILE}。请检查路径。")

    print(f"加载 BPE 分词器：{VOCAB_FILE}...")
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    with open(MERGES_FILE, 'rb') as f:
        merges = pickle.load(f)

    # 假设 special_tokens 至少包含 <|endoftext|>
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
    
    # 更新配置
    config.vocab_size = len(vocab)
    print(f"分词器加载成功，词汇表大小 (vocab_size): {config.vocab_size}")
    
    return tokenizer, config

# --- 辅助函数：加载和编码数据 (包含 np.memmap 逻辑) ---

def load_and_split_data(tokenizer: Tokenizer, config: TransformerConfig):
    """
    加载文本数据，使用 tokenizer 编码，并分割成训练/验证 NumPy 数组。
    如果 tokenized 文件已存在，则使用 np.load(mmap_mode='r') 内存映射加载。
    """
    # 指定 dtype 为 np.uint16，以匹配保存时的类型
    DATA_DTYPE = np.uint16 
    
    if os.path.exists(TOKENIZED_DATA_PATH):
        print(f"检测到预编码数据文件 {TOKENIZED_DATA_PATH}。使用内存映射 (mmap_mode='r') 加载。")
        # 核心修正：使用 mmap_mode='r'
        full_data = np.load(TOKENIZED_DATA_PATH, mmap_mode='r', allow_pickle=False)
        if full_data.dtype != DATA_DTYPE:
             print(f"警告：文件 dtype {full_data.dtype} 与预期 {DATA_DTYPE} 不匹配。")
    else:
        # --- 编码逻辑：将 List[int] 转换为 np.ndarray ---
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"未找到原始文本文件 {RAW_DATA_PATH}。请下载 TinyStories 数据集。")

        print(f"开始编码原始文本文件 {RAW_DATA_PATH}...")
        all_token_ids: List[int] = []
        
        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="编码进度"):
                ids = tokenizer.encode(line.strip())
                all_token_ids.extend(ids)

        # 核心转换：将 Python 列表转换为 NumPy 数组并保存
        full_data_in_mem = np.array(all_token_ids, dtype=DATA_DTYPE)
        
        # 保存 NumPy 数组 (np.save 会将 dtype 信息写入文件)
        np.save(TOKENIZED_DATA_PATH, full_data_in_mem)
        print(f"编码完成。Token 总数: {len(full_data_in_mem)}。已保存到 {TOKENIZED_DATA_PATH}")
        
        # 重新加载为内存映射模式
        full_data = np.load(TOKENIZED_DATA_PATH, mmap_mode='r', allow_pickle=False)

    # --- 分割数据 (内存映射数组可以直接分割) ---
    n = int(DATA_SPLIT_RATIO * len(full_data))
    
    data = {
        # data['train'] 和 data['val'] 都是 memory-mapped views of the file
        'train': full_data[:n], 
        'val': full_data[n:]    
    }
    
    print(f"数据分割完成：训练集 {len(data['train'])} 个 token，验证集 {len(data['val'])} 个 token。")
    return data

# --- 评估函数 (不变) ---

@torch.no_grad()
def estimate_loss(model: nn.Module, data: Dict[str, np.ndarray], eval_iters: int, device: str):
    model.eval()
    losses = torch.zeros(eval_iters)
    val_data = data['val']
    
    for k in range(eval_iters):
        X, Y = get_batch(val_data, BATCH_SIZE, MODEL_CONFIG.context_length, device)
        logits, _ = model(X)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
        
    model.train()
    return losses.mean().item()

# --- 主训练循环 ---

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 加载 Tokenizer, 配置和数据 (现在包含 mmap)
    tokenizer, config = load_tokenizer_and_update_config(MODEL_CONFIG)
    data = load_and_split_data(tokenizer, config) 

    # 2. 初始化模型和优化器
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        # rope_theta 保持默认值 10000，除非 config 中有明确设置
        # device, dtype 可以在 .to(device) 中处理
    ).to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )

    # 3. 处理检查点
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_step = 0
    # ... (如果需要恢复训练，请在此处设置检查点路径并调用 load_checkpoint) ...

    # 4. 训练循环
    model.train()
    print(f"从步骤 {start_step} 开始训练...")
    
    t0 = time.time()
    for step in range(start_step, MAX_STEPS):
        # A. 学习率调度
        lr = lr_cosine_schedule_with_warmup(step, MAX_STEPS, WARMUP_STEPS, LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # B. 数据加载 (从 memory-mapped array 中采样)
        X, Y = get_batch(data['train'], BATCH_SIZE, config.context_length, device)
        
        # C. 前向、损失、反向
        logits, _ = model(X)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) 
        
        optimizer.zero_grad()
        loss.backward()
        
        # F. 梯度裁剪
        gradient_clipping(model.parameters(), max_norm=GRAD_CLIP_NORM)
        
        optimizer.step()
        
        # G. 日志和评估
        if step % LOG_INTERVAL == 0 and step >= start_step:
            dt = time.time() - t0
            print(f"步骤 {step}: LR {lr:.2e}, Loss {loss.item():.4f}, Time/step {dt/LOG_INTERVAL*1000:.2f}ms")
            t0 = time.time()

        if step % EVAL_INTERVAL == 0 and step > start_step:
            val_loss = estimate_loss(model, data, eval_iters=200, device=device)
            val_ppl = math.exp(val_loss) 
            print(f"--- 评估 (步骤 {step}) ---")
            print(f"验证损失 (Val Loss): {val_loss:.4f}, 验证困惑度 (Val Perplexity): {val_ppl:.2f}")

        # H. 检查点保存
        if step % CHECKPOINT_INTERVAL == 0 and step > start_step:
            ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{step:06d}.pt')
            save_checkpoint(model, optimizer, step, ckpt_path)

    print(f"训练完成。")
    final_ckpt_path = os.path.join(checkpoint_dir, f'ckpt_{MAX_STEPS:06d}.pt')
    save_checkpoint(model, optimizer, MAX_STEPS, final_ckpt_path)


if __name__ == '__main__':
    train()