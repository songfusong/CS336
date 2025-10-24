import os
import pickle
import numpy as np
# --- 核心导入：使用 tqdm 的并行辅助函数 ---
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
# --- 核心导入：用于多进程设置 ---
import multiprocessing 
import math
from typing import Optional, List, Dict, Any, Tuple

# 假设您的 Tokenizer 模块位于 cs336_basics 路径下
try:
    from cs336_basics.tokenizer import Tokenizer
except ImportError:
    print("错误：无法导入 cs336_basics.tokenizer.Tokenizer。请确保路径和文件正确。")
    # 临时定义一个存根，确保代码结构完整
    class Tokenizer:
        def __init__(self, vocab, merges, special_tokens):
            self.id_to_token = vocab
            self.merges = merges
            self.special_tokens_set = set(special_tokens)
        def encode(self, text: str) -> List[int]: return [0] # 存根实现


# --- 训练配置常量 (用于数据加载) ---
DATA_SPLIT_RATIO = 0.9   
DATA_DTYPE = np.uint16   

# --- 文件路径 ---
VOCAB_FILE = 'cs336_basics/TinyStoriesTrainToken_vocab.pkl'
MERGES_FILE = 'cs336_basics/TinyStoriesTrainToken_merges.pkl'
RAW_DATA_PATH = 'data/TinyStoriesV2-GPT4-train.txt' 
TOKENIZED_DATA_PATH = 'train/tinystories_train_tokens.npy' # 我们将使用 uint16 dtype



# --- 辅助类 (用于配置信息) ---
class TransformerConfig:
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, dropout):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

# 默认配置 (用于演示)
MODEL_CONFIG = TransformerConfig(
    vocab_size=None, context_length=128, num_layers=6, 
    d_model=384, num_heads=6, d_ff=384 * 4, dropout=0.0
)

# --- 辅助函数：Worker 任务 (需要一个单独的函数来接受一个元组参数) ---
# process_map 期望 worker 函数只接受一个参数，但该参数可以是包含所有数据的元组。

def _encode_story_worker_args(args: Tuple[str, Dict[int, bytes], List[Tuple[bytes, bytes]], List[str]]) -> List[int]:
    """
    Worker function: 在子进程中编码单个故事，接受单个元组作为参数。
    """
    story_line, vocab, merges, special_tokens = args
    try:
        # 必须在子进程中重新实例化 Tokenizer
        worker_tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
        return worker_tokenizer.encode(story_line.strip())
    except Exception as e:
        # 打印到 stderr，以便在多进程环境中看到错误
        print(f"Worker 进程中编码失败: {e}", flush=True)
        return []

# --- 辅助函数：加载分词器 (不变) ---

def load_tokenizer_and_update_config(config: TransformerConfig):
    # ... (代码与前文相同，省略以保持简洁性) ...
    if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
        raise FileNotFoundError(f"未找到 BPE 文件：{VOCAB_FILE} 或 {MERGES_FILE}。请检查路径。")

    print(f"加载 BPE 分词器：{VOCAB_FILE}...")
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    with open(MERGES_FILE, 'rb') as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
    config.vocab_size = len(vocab)
    print(f"分词器加载成功，词汇表大小 (vocab_size): {config.vocab_size}")
    
    return tokenizer, config


# --- 核心函数：加载、编码、分割 (使用 tqdm.contrib.concurrent.process_map) ---

def load_and_split_data_optimized_visual(tokenizer: Tokenizer, config: TransformerConfig) -> Dict[str, np.ndarray]:
    """
    使用并行编码 (带可视化进度条) 和内存映射加载数据。
    """
    
    if os.path.exists(TOKENIZED_DATA_PATH):
        # ... (内存映射加载逻辑不变) ...
        print(f"检测到预编码数据文件 {TOKENIZED_DATA_PATH}。使用内存映射 (mmap_mode='r') 加载。")
        full_data = np.load(TOKENIZED_DATA_PATH, mmap_mode='r', allow_pickle=False)
    else:
        # --- 编码逻辑：使用 process_map 加速和可视化 ---
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"未找到原始文本文件 {RAW_DATA_PATH}。请下载 TinyStories 数据集。")

        print(f"开始并行编码原始文本文件 {RAW_DATA_PATH}...")
        
        # 1. 读取所有行到内存
        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # 2. 准备并行参数
        vocab = tokenizer.id_to_token 
        merges = tokenizer.merges     
        special_tokens = list(tokenizer.special_tokens_set)
        
        num_cpus = 2
        print(f"使用 {num_cpus} 个核心进行并行编码 (tqdm process_map)...")

        # 3. 准备所有任务参数列表
        # 每个任务是一个元组，包含 worker 函数所需的所有数据
        task_args = [
            (line, vocab, merges, special_tokens) for line in all_lines
        ]
        
        # 4. 使用 process_map 执行并行编码
        # process_map 自动创建和管理进程池，并提供进度条。
        # chunksize=1 表示一次只给每个进程发送一个元素，对于大型任务通常是高效的默认值。
        results_list_of_ids: List[List[int]] = process_map(
            _encode_story_worker_args, 
            task_args, 
            max_workers=num_cpus,
            chunksize=1, # 建议在 CPU 密集型任务中使用 1
            desc="并行编码进度 (故事/行)"
        )

        # 5. 收集结果 (将 List[List[int]] 展平为 List[int])
        all_token_ids: List[int] = [token_id for sublist in results_list_of_ids for token_id in sublist]

        # 6. 核心转换、保存和重新加载为内存映射
        full_data_in_mem = np.array(all_token_ids, dtype=DATA_DTYPE)
        
        np.save(TOKENIZED_DATA_PATH, full_data_in_mem)
        print(f"编码完成。Token 总数: {len(full_data_in_mem)}。已保存到 {TOKENIZED_DATA_PATH}")
        full_data = np.load(TOKENIZED_DATA_PATH, mmap_mode='r', allow_pickle=False)

    # --- 7. 分割数据 ---
    n = int(DATA_SPLIT_RATIO * len(full_data))
    
    data = {'train': full_data[:n], 'val': full_data[n:]}
    
    print(f"数据分割完成：训练集 {len(data['train'])} 个 token，验证集 {len(data['val'])} 个 token。")
    return data


# --- 演示运行区块 ---
if __name__ == '__main__':
    # 必须设置 'spawn' 启动方式以确保多进程环境的稳定性和安全性
    multiprocessing.set_start_method('spawn', force=True)
    
    print("--- 1. 加载 Tokenizer 并更新配置 ---")
    try:
        tokenizer, config = load_tokenizer_and_update_config(MODEL_CONFIG)
    except FileNotFoundError as e:
        print(f"无法继续：{e}")
        exit()

    print("\n--- 2. 加载和分割数据 (使用 tqdm.contrib.concurrent.process_map) ---")
    try:
        # 这个调用将在控制台显示一个带有并行化信息的进度条
        data_splits = load_and_split_data_optimized_visual(tokenizer, config)
    except Exception as e:
        print(f"\n数据加载过程中发生错误: {e}")
        exit()

    # ... (验证代码) ...
    print("\n--- 3. 验证加载结果 ---")
    train_data = data_splits['train']
    print(f"训练集 Token 总数: {len(train_data)}")