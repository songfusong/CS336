import regex as re
import collections
import os
import io
import multiprocessing as mp
import argparse # 导入 argparse 模块
from typing import Dict, List, Tuple, Generator, BinaryIO
import pickle
import base64

# --- 1. 预分词正则模式 (Pre-tokenization Regex) ---
# GPT-2/tiktoken 风格的预分词模式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# --- 2. 文件分块边界查找函数 ---
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently by finding a
    safe boundary (the split_special_token) near the desired position.
    
    Args:
        file: The opened file handle (BinaryIO).
        desired_num_chunks: The target number of chunks.
        split_special_token: The byte sequence to use as a safe split boundary.

    Returns:
        list[int]: A sorted list of unique byte offsets (boundaries).
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    # 查找文件总大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 计算每个分块的理论大小
    chunk_size = file_size // desired_num_chunks

    # 均匀分布的初始边界猜测
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # 确保最后一个边界是文件末尾

    mini_chunk_size = 4096  # 每次向前查找 4KB

    # 遍历除了起始和结束（0 和 file_size）以外的所有边界猜测
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从边界猜测位置开始

        # 尝试向前查找安全边界 (split_special_token)
        current_read_position = initial_position
        
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小块
            
            # 遇到文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到安全边界：当前读取位置 + 查找到的偏移量
                chunk_boundaries[bi] = current_read_position + found_at
                break
            
            # 更新当前读取位置并继续向前
            current_read_position += mini_chunk_size

    # 确保所有边界是唯一的，并保持排序
    return sorted(list(set(chunk_boundaries)))

# --- 3. 并行计数任务函数 ---

def count_token_pairs(
    input_path: str,
    start_offset: int,
    end_offset: int,
    # 接收特殊 token 字符串列表，用于分割文本
    special_tokens: List[str] 
) -> Tuple[Dict[Tuple[bytes, ...], int], Dict[Tuple[bytes, bytes], int]]:
    """
    处理文件的一个块，执行预分词，并计算初始 token 序列和相邻 token 对的频率。
    在应用预分词正则之前，使用特殊 token 作为分隔符进行分割。

    返回: (pre_tokens_counts, pair_counts)
    """
    
    pre_tokens_counts = collections.defaultdict(int)
    pair_counts = collections.defaultdict(int)
    
    # 以字节模式读取文件块
    with open(input_path, 'rb') as f:
        f.seek(start_offset)
        # 读取块内容
        block_bytes = f.read(end_offset - start_offset)
    
    # 将字节块解码为字符串进行正则预分词
    block_text = block_bytes.decode("utf-8", errors="replace")
    
    # 1. 使用特殊 token 作为分隔符进行分割
    if special_tokens:
        # 确保对特殊 token 字符串进行转义，并用括号捕获它们，以便 re.split 将它们保留在结果中。
        special_tokens_pattern = "|".join(re.escape(t) for t in special_tokens)
        text_chunks = re.split(f"({special_tokens_pattern})", block_text)
    else:
        # 如果没有特殊 token，则整个块都是一个 chunk
        text_chunks = [block_text]
    
    # 2. 对每个非特殊 token 块进行预分词和计数
    for chunk in text_chunks:
        # 检查 chunk 是否是特殊 token 字符串本身 (这些 token 不参与 BPE 合并训练)
        if chunk in special_tokens:
            continue
            
        # 对非特殊 token 的文本块执行预分词
        for match in re.finditer(PAT, chunk):
            pre_token_bytes = match.group(0).encode("utf-8")
            
            # 一个 pre-token 最初是其组成字节的序列
            initial_byte_sequence = tuple(bytes([b]) for b in pre_token_bytes)
            
            # 统计 pre-token 序列的频率
            pre_tokens_counts[initial_byte_sequence] += 1
            
            # 统计相邻 token 对的频率
            for i in range(len(initial_byte_sequence) - 1):
                pair = (initial_byte_sequence[i], initial_byte_sequence[i+1])
                pair_counts[pair] += 1
            
    return pre_tokens_counts, pair_counts

# --- 4. BPE 合并循环辅助函数 ---

def find_best_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes] | None:
    """根据频率和词典序确定最佳合并对"""
    if not pair_counts:
        return None
    
    # 1. 找到最高频率
    max_freq = max(pair_counts.values())
    
    # 2. 筛选出所有最高频率的对
    best_pairs = {pair: freq for pair, freq in pair_counts.items() if freq == max_freq}
    
    # 3. 确定性地打破平局：偏向**词典序最大**的对
    # max() 对元组进行比较时就是按词典序比较
    return max(best_pairs.keys())

def merge_pair(token_sequence: Tuple[bytes, ...], pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
    """在一个 pre-token 序列中执行合并操作"""
    merged_sequence = []
    i = 0
    N = len(token_sequence)
    while i < N:
        # 检查是否有匹配的对
        if i < N - 1 and (token_sequence[i], token_sequence[i+1]) == pair:
            merged_sequence.append(new_token)
            i += 2
        else:
            merged_sequence.append(token_sequence[i])
            i += 1
    return tuple(merged_sequence)

# --- 5. 主训练函数 ---

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 4 # 默认使用 4 个进程进行并行
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练一个字节级 BPE 分词器，使用并行化进行初始计数。

    参数:
        input_path: str, BPE 训练数据文本文件的路径。
        vocab_size: int, 最终词汇表的最大大小。
        special_tokens: list[str], 要添加到词汇表中的特殊 token 字符串列表。
        num_processes: int, 用于并行处理的进程数。

    返回:
        vocab: dict[int, bytes], 分词器词汇表，从 ID 到 token 字节序列的映射。
        merges: list[tuple[bytes, bytes]], BPE 合并操作列表，按创建顺序排列。
    """
    
    print(f"--- BPE Tokenizer Training Started ---")
    print(f"Target Vocab Size: {vocab_size}")

    # --- A. 初始化词汇表 (Byte Vocabulary + Special Tokens) ---
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    
    # 将特殊 token 添加到词汇表
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_token_id] = token_bytes
            next_token_id += 1

    num_merges = vocab_size - next_token_id
    merges = []

    if num_merges <= 0:
        print(f"Initial vocab size {next_token_id} already meets or exceeds target {vocab_size}. Stopping.")
        return vocab, merges
    
    print(f"Number of merges to perform: {num_merges}")

    # --- B. 并行初始计数 (Parallel Initial Counting) ---
    print(f"Finding safe chunk boundaries using {num_processes} processes...")
    
    # 假设使用第一个特殊 token（如果存在）作为分块的安全边界。
    # 如果没有特殊 token，则使用一个不可能出现在预分词 token 中的字节序列，
    # 例如 b'\x00' (NUL 字节) 或 b'\n' (换行符)。这里默认使用 b'\n'。
    split_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b'\n'
    
    # 打开文件并查找边界
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)

    # 准备并行任务的参数列表
    tasks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if start < end: # 避免空块
            # 传递 special_tokens (List[str]) 字符串列表
            tasks.append((input_path, start, end, special_tokens))
    
    # 使用进程池执行并行计数
    pool = mp.Pool(processes=len(tasks)) # 使用实际任务数作为进程数
    # results 包含 (pre_tokens_counts, pair_counts) 的元组列表
    results = pool.starmap(count_token_pairs, tasks)
    pool.close()
    pool.join()
    
    # 聚合结果
    pre_tokens_counts = collections.defaultdict(int)
    pair_counts = collections.defaultdict(int)
    
    for pre_counts, pair_c in results:
        # 聚合 pre-token 序列计数
        for seq, count in pre_counts.items():
            pre_tokens_counts[seq] += count
        # 聚合相邻对计数
        for pair, count in pair_c.items():
            pair_counts[pair] += count
    
    print(f"Initial counting complete. Found {len(pre_tokens_counts)} unique pre-token sequences.")
    
    # --- C. BPE 合并循环 (Sequential Merging Loop) ---
    
    for merge_step in range(num_merges):
        # 1. 找到最频繁的对
        best_pair = find_best_pair(pair_counts)
        
        if best_pair is None:
            print(f"No more pairs to merge. Stopping at step {merge_step}.")
            break
            
        token1, token2 = best_pair
        
        # 2. 创建新的合并 token
        new_token = token1 + token2

        # 3. 更新 merges 列表和 vocab 字典
        merges.append(best_pair)
        vocab[next_token_id] = new_token
        new_token_id = next_token_id
        next_token_id += 1
        
        # 4. 更新 pre-token 序列和相邻对频率
        
        # 追踪受影响的 token 序列，以便后续更新 pair_counts
        affected_sequences = [] 
        
        # 用于存储更新后的 pre_tokens_counts
        new_pre_tokens_counts = collections.defaultdict(int)

        # 遍历所有 pre-token 序列
        items_to_process = list(pre_tokens_counts.items())
        
        for token_sequence, freq in items_to_process:
            
            # 检查序列是否包含最佳对
            if len(token_sequence) >= 2 and any(token_sequence[i:i+2] == best_pair for i in range(len(token_sequence) - 1)):
                
                # 如果包含，则执行合并
                new_sequence = merge_pair(token_sequence, best_pair, new_token)
                
                # 将旧序列的频率转移到新序列
                new_pre_tokens_counts[new_sequence] += freq
                
                # 记录旧序列（将被移除）和新序列（将被添加）以更新 pair_counts
                affected_sequences.append((token_sequence, new_sequence, freq))
            else:
                # 不包含该对的序列保持不变 (直接转移到新字典)
                new_pre_tokens_counts[token_sequence] += freq
        
        # 更新 pre_tokens_counts 为新字典
        pre_tokens_counts = new_pre_tokens_counts

        # 5. **增量更新 pair_counts** (高效的关键步骤)
        
        # 对于每个受影响的 token 序列，我们从 pair_counts 中减去旧对的计数，并加上新对的计数。
        for old_seq, new_seq, freq in affected_sequences:
            # 移除旧对的计数
            for i in range(len(old_seq) - 1):
                pair_counts[(old_seq[i], old_seq[i+1])] -= freq
                # 清除计数为 0 的对
                if pair_counts[(old_seq[i], old_seq[i+1])] == 0:
                    del pair_counts[(old_seq[i], old_seq[i+1])]

            # 添加新对的计数
            for i in range(len(new_seq) - 1):
                pair_counts[(new_seq[i], new_seq[i+1])] += freq

        # 打印进度
        if (merge_step + 1) % 1000 == 0:
            print(f"Step {merge_step + 1}/{num_merges}: Merged {token1!r} + {token2!r} -> {new_token!r}")

    print(f"--- BPE Training Finished. Final Vocab Size: {next_token_id} ---")
    return vocab, merges

# --- 6. 命令行主函数和示例用法 ---

def main():
    """
    解析命令行参数并执行 BPE 分词器训练，并将结果保存到磁盘。
    """
    parser = argparse.ArgumentParser(description="Train a Byte-Level BPE Tokenizer.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="bpe_training_data.txt",
        help="Path to the corpus file for training. Default: bpe_training_data.txt",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=300,
        help="Target size of the final vocabulary (including bytes and special tokens). Default: 300",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        default="<PAD>,<SEP>",
        help="Comma-separated list of special tokens to include. Default: <PAD>,<SEP>",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes to use for parallel initial counting. Default: 16",
    )
    # 新增参数：用于指定输出文件路径
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="bpe_tokenizer",
        help="Prefix for the output files (e.g., 'my_tokenizer' will result in my_tokenizer_vocab.pkl and my_tokenizer_merges.pkl). Default: bpe_tokenizer",
    )

    args = parser.parse_args()

    # 1. 处理特殊 token 列表
    special_tokens_list = [t.strip() for t in args.special_tokens.split(',') if t.strip()]
    
    # 2. 运行训练函数
    final_vocab, final_merges = train_bpe_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens_list,
        num_processes=args.num_processes
    )
    
    # 3. 打印结果摘要
    print("\n--- Training Results Summary ---")
    print(f"Total Merges Performed: {len(final_merges)}")
    print(f"Final Vocab Size: {len(final_vocab)}")
    print(f"First 5 Merges: {final_merges[:5]!r}") # 使用 !r 确保字节序列打印清晰
    
    # 查找并打印特殊 token 的 ID
    special_token_bytes = {t.encode('utf-8') for t in special_tokens_list}
    special_token_map = {v.decode('utf-8'): k for k, v in final_vocab.items() if v in special_token_bytes}
    print(f"Special Tokens IDs: {special_token_map}")
    
    # 查找并打印新创建的 token
    new_token_start_id = 256 + len(special_token_bytes)
    new_tokens = {k: v.decode('utf-8', errors='ignore') for k, v in final_vocab.items() if k >= new_token_start_id}
    print(f"Top 5 New Merged Tokens: {list(new_tokens.items())[:5]}")
    
    # --- 新增步骤：序列化保存结果 ---
    vocab_file = f"{args.output_prefix}_vocab.pkl"
    merges_file = f"{args.output_prefix}_merges.pkl"
    
    print("\n--- Saving Tokenizer Results ---")
    
    try:
        # 使用 pickle 序列化并保存词汇表（字典）
        with open(vocab_file, 'wb') as f:
            pickle.dump(final_vocab, f)
        print(f"✅ 词汇表 (vocab) 已保存到: {vocab_file}")

        # 使用 pickle 序列化并保存合并操作列表
        with open(merges_file, 'wb') as f:
            pickle.dump(final_merges, f)
        print(f"✅ 合并操作 (merges) 已保存到: {merges_file}")
        
    except Exception as e:
        print(f"❌ 警告：保存文件失败: {e}")


if __name__ == '__main__':
    main()
