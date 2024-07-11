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
SpecialTokenToIndex = dict[str, int]
IndexToSpecialToken = dict[int, str]
BytePairStats = dict[TokenPair, int]
EncodedString = list[int]
raw_bytes = 256


class BytePairEncoding:
    """Simple implementation of the BPE tokenizer that actually works on byte pairs."""

    vocab_size: int
    special_tokens: list[str]
    use_regex: bool
    _pair_to_index: Optional[PairToIndex] = None
    _index_to_pair: Optional[IndexToPair] = None
    _special_token_to_index: SpecialTokenToIndex = dict()
    _index_to_special_token: IndexToSpecialToken = dict()

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
        assert (
            self.vocab_size - raw_bytes == len(pti)
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
        assert (
            self.vocab_size - raw_bytes == len(itp)
        ), f"Dictionary expected to contain {self.vocab_size - raw_bytes}, got {len(itp)}"
        self._index_to_pair = itp
        self._pair_to_index = utils.invert_dict(itp)

    def __init__(
        self, vocab_size: int, special_tokens: list[str] = [], use_regex: bool = True
    ):
        assert vocab_size > raw_bytes + len(
            special_tokens
        ), f"Vocab size must be at least {raw_bytes + len(special_tokens)}"
        self.vocab_size = vocab_size
        self.use_regex = use_regex

        for i in range(len(special_tokens)):
            ix = self.vocab_size - len(special_tokens) + i
            self._special_token_to_index[special_tokens[i]] = ix
            self._index_to_special_token[ix] = special_tokens[i]

    @staticmethod
    def from_pretrained(
        pair_to_index: PairToIndex, use_regex: bool = True
    ) -> BytePairEncoding:
        bpe = BytePairEncoding(raw_bytes + len(pair_to_index), use_regex=use_regex)
        bpe.pair_to_index = pair_to_index
        return bpe

    def _separate_special_tokens(self, text: str) -> list[str]:
        special_pattern = f"""({"|".join(re.escape(tok) for tok in self._special_token_to_index.keys())})"""
        return re.split(special_pattern, text)

    # TODO: Optimize. Keep track of positions of pairs. When merging - simultaniously count new pairs, while subtracting from the previous ones.
    def train(
        self, text: str, save_path: Optional[str] = None, show_progress: bool = False
    ) -> PairToIndex:
        self._pair_to_index = dict()

        # Separate special tokens
        candidate_texts: list[str] = [text]
        if len(self._special_token_to_index) != 0:
            candidate_texts = self._separate_special_tokens(text)

        # Separate parts with no special tokens
        texts: list[str] = []
        if self.use_regex:
            pattern = re.compile(self._GPT4_SPLIT_PATTERN)
            for t in candidate_texts:
                if t in self._special_token_to_index.keys():
                    continue  # no need to train on special tokens
                texts.extend(pattern.findall(t))

        if len(texts) == 0:
            texts = [text]

        parts_bytes: list[EncodedString] = list(
            map(lambda part: list(part.encode("utf-8")), texts)
        )
        rng = range(raw_bytes, self.vocab_size - len(self._special_token_to_index), 1)
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
            elif i in self._index_to_special_token.keys():
                decoded_bytes += self._index_to_special_token[i].encode("utf-8")
            else:
                decoded_bytes += self._decode(list(self.index_to_pair[i]))
        return decoded_bytes

    def decode(self, encoded_text: EncodedString) -> str:
        return self._decode(encoded_text).decode("utf-8", errors="replace")

    def encode(self, text: str) -> EncodedString:
        text_parts = [text]
        if len(self._special_token_to_index) != 0:
            text_parts = self._separate_special_tokens(text)

        encoded_text: EncodedString = list()
        for part in text_parts:
            # if the text part is special token, process it
            if part in self._special_token_to_index.keys():
                encoded_text.append(self._special_token_to_index[part])
                continue

            text_bytes: list[EncodedString] = [list(part.encode("utf-8"))]
            while len(text_bytes[0]) >= 2:
                stats = self._get_stats(text_bytes)
                pair = min(stats, key=lambda p: self.pair_to_index.get(p, float("inf")))
                if pair not in self.pair_to_index:
                    break
                text_bytes = self._merge(text_bytes, pair, self.pair_to_index[pair])
            encoded_text.extend(text_bytes[0])
        return encoded_text

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
