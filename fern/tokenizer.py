from __future__ import annotations
import pathlib
from typing import Optional
import json
import regex as re
import tqdm.auto as tqdm
from fern import utils

TokenPair = tuple[int, int]
PairToIndex = dict[TokenPair, int]
IndexToPair = dict[int, TokenPair]
BytePairStats = dict[TokenPair, int]
EncodedString = list[int]
raw_bytes = 256


class BytePairEncoding:
    """Simple implementation of the BPE tokenizer that actually works on byte pairs."""

    vocab_size: int
    use_regex: bool
    _pair_to_index: Optional[PairToIndex] = None
    _index_to_pair: Optional[IndexToPair] = None

    _GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    @property
    def pair_to_index(self) -> PairToIndex:
        if self._pair_to_index is None:
            raise ValueError(
                "`pair_to_index` is not initialized. Consider training tokenizer."
            )
        return self._pair_to_index

    @pair_to_index.setter
    def pair_to_index(self, pti: PairToIndex):
        assert self.vocab_size - raw_bytes == len(
            pti
        ), f"Dictionary expected to contain {self.vocab_size - raw_bytes}, got {len(pti)}"
        self._pair_to_index = pti
        self._index_to_pair = utils.invert_dict(pti)

    @property
    def index_to_pair(self) -> IndexToPair:
        if self._index_to_pair is None:
            raise ValueError(
                "`index_to_pair` is not initialized. Consider training tokenizer."
            )
        return self._index_to_pair

    @index_to_pair.setter
    def index_to_pair(self, itp: IndexToPair):
        assert self.vocab_size - raw_bytes == len(
            itp
        ), f"Dictionary expected to contain {self.vocab_size - raw_bytes}, got {len(itp)}"
        self._index_to_pair = itp
        self._pair_to_index = utils.invert_dict(itp)

    def __init__(self, vocab_size: int, use_regex: bool = True):
        assert vocab_size > raw_bytes, "Vocab size must be at least 257"
        self.vocab_size = vocab_size
        self.use_regex = use_regex

    @staticmethod
    def from_pretrained(
        pair_to_index: PairToIndex, use_regex: bool = True
    ) -> BytePairEncoding:
        bpe = BytePairEncoding(raw_bytes + len(pair_to_index), use_regex=use_regex)
        bpe.pair_to_index = pair_to_index
        return bpe

    # TODO: Optimize. Keep track of positions of pairs. When merging - simultaniously count new pairs, while subtracting from the previous ones.
    def train(
        self, text: str, save_path: Optional[str] = None, show_progress: bool = False
    ) -> PairToIndex:
        self._pair_to_index = dict()
        texts = [text]
        if self.use_regex:
            pattern = re.compile(self._GPT4_SPLIT_PATTERN)
            texts = pattern.findall(text)
        parts_bytes: list[EncodedString] = list(
            map(lambda part: list(part.encode("utf-8")), texts)
        )
        rng = range(raw_bytes, self.vocab_size, 1)
        if show_progress:
            rng = tqdm.tqdm(rng)
        for i in rng:
            stats = self._get_stats(parts_bytes)
            frequent_pair = max(stats, key=lambda x: stats[x])
            parts_bytes = self._merge(parts_bytes, frequent_pair, i)
            self.pair_to_index[frequent_pair] = i
        self._index_to_pair = dict((v, k) for k, v in self.pair_to_index.items())
        if save_path is not None:
            self.save(save_path)
        return self.pair_to_index

    def _get_stats(self, parts_bytes: list[EncodedString]) -> BytePairStats:
        pair_counts: BytePairStats = dict()
        for part in parts_bytes:
            for pair in zip(part, part[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def _merge(
        self, parts_bytes: list[EncodedString], pair: TokenPair, token_index: int
    ) -> list[EncodedString]:
        merged_text: list[EncodedString] = []
        for part in parts_bytes:
            i = 0
            part_merge: EncodedString = []
            while i < len(part):
                if i != len(part) - 1 and part[i] == pair[0] and part[i + 1] == pair[1]:
                    part_merge.append(token_index)
                    i += 2  # skip pair of tokens
                else:
                    part_merge.append(part[i])
                    i += 1
            # part_merge.append(part[-1])
            merged_text.append(part_merge)
        return merged_text

    def _decode(self, enocded_text: EncodedString) -> bytes:
        decoded_bytes = b""
        for i in enocded_text:
            if i < raw_bytes:
                decoded_bytes += bytes([i])
            else:
                decoded_bytes += self._decode(list(self.index_to_pair[i]))
        return decoded_bytes

    def decode(self, encoded_text: EncodedString) -> str:
        return self._decode(encoded_text).decode("utf-8", errors="replace")

    def encode(self, text: str) -> EncodedString:
        text_bytes: list[EncodedString] = [list(text.encode("utf-8"))]
        while len(text_bytes[0]) >= 2:
            stats = self._get_stats(text_bytes)
            pair = min(stats, key=lambda p: self.pair_to_index.get(p, float("inf")))
            if pair not in self.pair_to_index:
                break
            text_bytes = self._merge(text_bytes, pair, self.pair_to_index[pair])
        return text_bytes[0]

    def save(self, path: str):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(list(self.pair_to_index.items())))

    @staticmethod
    def load(path: str) -> BytePairEncoding:
        p = pathlib.Path(path)
        loaded_json: list[tuple[TokenPair, int]] = json.loads(p.read_bytes())
        flat_pair_to_index: list[tuple[TokenPair, int]] = list(
            map(lambda x: ((x[0][0], x[0][1]), x[1]), loaded_json)
        )
        return BytePairEncoding.from_pretrained(dict(flat_pair_to_index))
