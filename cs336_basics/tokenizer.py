import re
import base64
from typing import Dict, List, Tuple, Iterable, Iterator, Union, Type, ClassVar

# --- 1. 预分词正则模式 (Pre-tokenization Regex) ---
# GPT-2/tiktoken 风格的预分词模式，用于将文本切分成逻辑块。
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\u00C0-\u017F]+| ?[0-9]+| ?[^\sA-Za-z0-9\u00C0-\u017F]+|\s+(?!\S)|\s+""",
    re.UNICODE 
)


class Tokenizer:
    """
    实现了支持特殊令牌和预分词的字节对编码(BPE)分词器。
    """
    
    # ----------------------------------------------------------------------
    # 构造函数
    # ----------------------------------------------------------------------
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Union[List[str], None] = None):
        """
        构造函数。
        """
        
        self.id_to_token: Dict[int, bytes] = vocab
        self.token_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.merges_map: Dict[Tuple[bytes, bytes], bytes] = {}

        # 1. 特殊令牌处理
        self.special_tokens_list: List[str] = special_tokens if special_tokens else []
        self.special_tokens_set: set = set(self.special_tokens_list)
        
        # 将特殊令牌添加到词汇表中，分配新的 ID
        next_id = max(self.id_to_token.keys()) + 1 if self.id_to_token else 0
        for s in self.special_tokens_list:
            token_bytes = s.encode('utf-8')
            if token_bytes not in self.token_to_id:
                self.token_to_id[token_bytes] = next_id
                self.id_to_token[next_id] = token_bytes
                next_id += 1
                
        # 2. 构建特殊令牌正则表达式 (按长度降序确保最长匹配优先)
        if self.special_tokens_list:
            sorted_tokens = sorted(list(self.special_tokens_set), key=len, reverse=True)
            escaped_tokens = [re.escape(s) for s in sorted_tokens]
            self.special_tokens_pattern = re.compile(f"({'|'.join(escaped_tokens)})")
        else:
            self.special_tokens_pattern = None

        # 3. 构建合并结果映射 (merges_map)
        for token1, token2 in merges:
            merged_token = token1 + token2
            if merged_token in self.token_to_id:
                self.merges_map[(token1, token2)] = merged_token

    # --------------------------------------------------------------------------
    # 核心 BPE 编码逻辑
    # --------------------------------------------------------------------------
    
    def _bpe_encode_bytes(self, token_bytes: bytes, 
                         forbidden_pairs: Union[set, None] = None) -> List[int]:
        """
        对原始字节序列执行 BPE 合并。
        """
        
        tokens: List[bytes] = [bytes([b]) for b in token_bytes]
        forbidden = forbidden_pairs if forbidden_pairs is not None else set() 
        
        while True:
            best_pair = None
            min_rank = float('inf')
            
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                
                # 检查禁止合并对
                if pair in forbidden: 
                    i += 1 
                    continue
                
                rank = self.merges_rank.get(pair)
                
                if rank is not None and rank < min_rank:
                    if pair in self.merges_map:
                        min_rank = rank
                        best_pair = pair
                i += 1
            
            if best_pair is None:
                break
                
            merged_token = self.merges_map[best_pair]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2 
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens 

        return [self.token_to_id[t] for t in tokens]

    # --------------------------------------------------------------------------
    # 公共接口方法
    # --------------------------------------------------------------------------
    
    @classmethod
    def from_files(cls: Type['Tokenizer'], vocab_filepath: str, merges_filepath: str, special_tokens: Union[List[str], None] = None) -> 'Tokenizer':
        """
        类方法。从序列化的词汇表和合并规则文件中构造并返回 Tokenizer。
        """
        
        vocab: Dict[int, bytes] = {}
        merges: List[Tuple[bytes, bytes]] = []

        try:
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) != 2: continue
                    
                    token_id = int(parts[0])
                    token_bytes = base64.b64decode(parts[1])
                    vocab[token_id] = token_bytes
        except Exception as e:
            raise IOError(f"从 {vocab_filepath} 加载词汇表时出错: {e}")

        try:
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) != 2: continue
                    
                    token1 = base64.b64decode(parts[0])
                    token2 = base64.b64decode(parts[1])
                    merges.append((token1, token2))
        except Exception as e:
            raise IOError(f"从 {merges_filepath} 加载合并规则时出错: {e}")
            
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """将输入文本编码为 token ID 序列。"""
        
        # 兼容性：确保只有在没有特殊令牌时才对整个文本应用 BPE
        if not self.special_tokens_pattern:
            token_bytes = text.encode('utf-8')
            return self._bpe_encode_bytes(token_bytes, forbidden_pairs=None)

        # 1. 特殊令牌切分
        parts = self.special_tokens_pattern.split(text)
        
        ids: List[int] = []
        for part in parts:
            if not part: continue
            
            if part in self.special_tokens_set:
                # 2. 如果是特殊令牌，直接编码
                token_bytes = part.encode('utf-8')
                ids.append(self.token_to_id[token_bytes])
            else:
                # 3. 核心修改：对普通文本块应用 PAT 预分词，使用 finditer 替代 findall
                pre_tokens: Iterator[re.Match] = PAT.finditer(part)
                
                for match in pre_tokens:
                    pre_token = match.group(0) # 从 Match 对象中提取匹配的字符串
                    token_bytes = pre_token.encode('utf-8')
                    # 4. 对每个预分词块执行 BPE 编码，不再应用禁止规则
                    ids.extend(self._bpe_encode_bytes(token_bytes, forbidden_pairs=None)) 
                    
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定一个字符串的可迭代对象，返回一个生成器，惰性地生成 token ID。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        """将 token ID 序列解码为文本。"""
        
        tokens: List[bytes] = []
        for token_id in ids:
            token_bytes = self.id_to_token.get(token_id)
            if token_bytes is not None:
                tokens.append(token_bytes)
            else:
                tokens.append(b'\xef\xbf\xbd') 
                
        return b''.join(tokens).decode('utf-8', errors='replace')
