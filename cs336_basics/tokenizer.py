import re
import base64
from typing import Dict, List, Tuple, Iterable, Iterator, Union, Type, ClassVar

class Tokenizer:
    """
    实现了支持特殊令牌的字节对编码(BPE)分词器。
    
    BPE 过程（在 self._bpe_encode_bytes 中）通过全局迭代地应用最高优先级
    的合并规则（来自 'merges' 列表），直到无法再进行合并。
    """
    
    # ----------------------------------------------------------------------
    # 构造函数: 包含 special_tokens 参数
    # ----------------------------------------------------------------------
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Union[List[str], None] = None):
        """
        构造函数。
        """
        
        # 核心数据结构 (保持不变)
        self.id_to_token: Dict[int, bytes] = vocab
        self.token_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.merges_map: Dict[Tuple[bytes, bytes], bytes] = {}

        # 1. 特殊令牌处理 (Special Tokens Handling)
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
                
        # 2. 构建正则表达式
        if self.special_tokens_list:
            # 核心修复: 按长度降序排序特殊令牌，以确保最长匹配优先
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
    # 核心 BPE 编码逻辑 (已修改为接受 forbidden_pairs)
    # --------------------------------------------------------------------------
    
    def _bpe_encode_bytes(self, token_bytes: bytes, 
                         forbidden_pairs: Union[set, None] = None) -> List[int]:
        """
        对原始字节序列执行 BPE 合并。
        接受一个可选的 forbidden_pairs 集合来禁止某些合并（用于兼容 tiktoken）。
        """
        
        # 1. 从初始 token 开始：单字节 token 列表
        tokens: List[bytes] = [bytes([b]) for b in token_bytes]
        
        # 使用传入的禁止对，或默认为空集
        forbidden = forbidden_pairs if forbidden_pairs is not None else set() 
        
        while True:
            best_pair = None
            min_rank = float('inf')
            
            # 2. 查找当前 token 序列中可应用的最高优先级（排名最低）合并规则
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                
                # ==== 修复: 检查传入的禁止合并对 ====
                if pair in forbidden: 
                    i += 1 
                    continue
                # =======================================
                
                rank = self.merges_rank.get(pair)
                
                if rank is not None and rank < min_rank:
                    # 检查合并结果是否在我们的映射中（即它是否有 ID）
                    if pair in self.merges_map:
                        min_rank = rank
                        best_pair = pair
                i += 1
            
            # 3. 如果没有找到可用的合并规则，退出循环
            if best_pair is None:
                break
                
            # 4. 将最高优先级的合并规则全局应用到序列中的所有出现位置
            merged_token = self.merges_map[best_pair]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 检查是否匹配到最佳合并对并进行合并
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2 # 跳过被合并的第二个 token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens # 更新序列以进行下一轮迭代

        # 5. 将最终的字节 token 映射为它们的 ID
        return [self.token_to_id[t] for t in tokens]

    # --------------------------------------------------------------------------
    # 公共接口方法 (已修改 encode 逻辑)
    # --------------------------------------------------------------------------
    
    @classmethod
    def from_files(cls: Type['Tokenizer'], vocab_filepath: str, merges_filepath: str, special_tokens: Union[List[str], None] = None) -> 'Tokenizer':
        """
        类方法。从序列化的词汇表和合并规则文件中构造并返回 Tokenizer。
        """
        
        vocab: Dict[int, bytes] = {}
        merges: List[Tuple[bytes, bytes]] = []

        # 加载词汇表：格式 'ID BASE64_ENCODED_BYTES'
        try:
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) != 2: continue
                    
                    token_id = int(parts[0])
                    # 从 Base64 字符串解码回 bytes
                    token_bytes = base64.b64decode(parts[1])
                    vocab[token_id] = token_bytes
        except Exception as e:
            raise IOError(f"从 {vocab_filepath} 加载词汇表时出错: {e}")

        # 加载合并规则：格式 'BASE64_ENCODED_BYTES1 BASE64_ENCODED_BYTES2'
        try:
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) != 2: continue
                    
                    # 从 Base64 字符串解码回 bytes
                    token1 = base64.b64decode(parts[0])
                    token2 = base64.b64decode(parts[1])
                    merges.append((token1, token2))
        except Exception as e:
            raise IOError(f"从 {merges_filepath} 加载合并规则时出错: {e}")
            
        # 传入 special_tokens 参数
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """将输入文本编码为 token ID 序列。"""
        
        # 定义 GPT-2/GPT-3 兼容模式下常用的禁止合并对
        DEFAULT_FORBIDDEN = {(b'\n', b'\n'), (b'\r', b'\n')}
        
        if not self.special_tokens_pattern:
            # 情况 1: 没有特殊令牌，对整个文本的字节序列执行 BPE 编码，应用禁止规则
            token_bytes = text.encode('utf-8')
            return self._bpe_encode_bytes(token_bytes, forbidden_pairs=DEFAULT_FORBIDDEN)

        # 情况 2: 存在特殊令牌。使用正则表达式切分文本。
        parts = self.special_tokens_pattern.split(text)
        
        ids: List[int] = []
        for part in parts:
            if not part:
                continue
            
            if part in self.special_tokens_set:
                # 切分部分是特殊令牌
                token_bytes = part.encode('utf-8')
                ids.append(self.token_to_id[token_bytes])
            else:
                # 切分部分是普通文本，应用 BPE 编码
                token_bytes = part.encode('utf-8')
                
                # 兼容性修复：如果文本块是纯粹的换行符 \n\n 或 \r\n，则不应用禁止规则
                # 这是因为 tiktoken 在特殊令牌边界允许这些特殊的合并。
                forbidden_to_use = DEFAULT_FORBIDDEN
                if part in ('\n\n', '\r\n'):
                    forbidden_to_use = set() # 禁用禁止规则
                    
                ids.extend(self._bpe_encode_bytes(token_bytes, forbidden_pairs=forbidden_to_use))
                
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
                # 处理未知 ID（词汇表外）
                tokens.append(b'\xef\xbf\xbd') # 使用 UTF-8 替换字符
                
        # 连接所有字节 token，并使用 'replace' 错误处理解码为字符串
        return b''.join(tokens).decode('utf-8', errors='replace')