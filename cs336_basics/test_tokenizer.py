import re
import base64
from typing import Dict, List, Tuple, Iterable, Iterator, Union, Type, ClassVar
import pickle
from cs336_basics.tokenizer import Tokenizer

def load_bpe_data(vocab_filepath: str, merges_filepath: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    从 .pkl 文件加载 GPT-2 风格的 BPE 词汇表和合并规则。

    Args:
        vocab_filepath: 词汇表文件的路径 (e.g., bpe_tokenizer_vocab.pkl)
        merges_filepath: 合并规则文件的路径 (e.g., bpe_tokenizer_merges.pkl)

    Returns:
        一个包含 (vocab, merges) 的元组。
        vocab: Dict[int, bytes]，将 token ID 映射到字节序列。
        merges: List[Tuple[bytes, bytes]]，按优先级排序的字节对合并列表。
    """
    vocab: Dict[int, bytes] = {}
    merges: List[Tuple[bytes, bytes]] = []

    # 1. 加载词汇表 (vocab)
    try:
        with open(vocab_filepath, 'rb') as f:
            # 假设 vocab.pkl 存储的是 Dict[int, bytes]
            # 注意: 'rb' 表示以二进制模式读取
            loaded_vocab = pickle.load(f)
            if isinstance(loaded_vocab, dict):
                # 假设 pickle 文件的内容是 {ID: token_bytes}
                vocab = loaded_vocab
            else:
                raise TypeError("Vocab file does not contain a dictionary.")

    except FileNotFoundError:
        raise FileNotFoundError(f"词汇表文件未找到: {vocab_filepath}")
    except Exception as e:
        raise IOError(f"加载词汇表时发生 pickle 错误: {e}")

    # 2. 加载合并规则 (merges)
    try:
        with open(merges_filepath, 'rb') as f:
            # 假设 merges.pkl 存储的是 List[Tuple[bytes, bytes]]
            loaded_merges = pickle.load(f)
            if isinstance(loaded_merges, list):
                # 假设 pickle 文件的内容是 [(token1_bytes, token2_bytes), ...]
                merges = loaded_merges
            else:
                raise TypeError("Merges file does not contain a list.")

    except FileNotFoundError:
        raise FileNotFoundError(f"合并规则文件未找到: {merges_filepath}")
    except Exception as e:
        raise IOError(f"加载合并规则时发生 pickle 错误: {e}")

    return vocab, merges

def test_encode_special_token_trailing_newlines():
    merges_path = "cs336_basics/TinyStoriesTrainToken_merges.pkl"
    vocab_path = "cs336_basics/TinyStoriesTrainToken_vocab.pkl"
    special_tokens=["<|endoftext|>"]
    
    vocab, merges = load_bpe_data(vocab_filepath=vocab_path, merges_filepath=merges_path)

    corpus_path = "tests/fixtures/tinystories_sample.txt"
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    with open(corpus_path) as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    print(ids)
    print(tokenizer.decode(ids))

def main():
    test_encode_special_token_trailing_newlines()

if __name__ == "__main__": # 注意：是双下划线，不是单下划线
    main()

#哇，我真的训练了一个tokenizer！