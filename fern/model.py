import typing as t

import torch
import torchtune  # type: ignore

from fern.config import FernConfig


class SelfAttentionHead(torch.nn.Module):
    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.key = torch.nn.Linear(
            self.config.d_model, self.config.head_size, bias=False
        )
        self.query = torch.nn.Linear(
            self.config.d_model, self.config.head_size, bias=False
        )
        self.value = torch.nn.Linear(
            self.config.d_model, self.config.head_size, bias=False
        )
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(self.config.block_size, self.config.block_size)),
        )
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _B, T, _C = x.size()
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)

        # compute attention scores
        wei = (
            q @ k.transpose(-2, -1) * k.size(-1) ** -0.5
        )  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -torch.inf)  # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)

        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.heads = torch.nn.ModuleList(
            [SelfAttentionHead(self.config) for _ in range(self.config.n_heads)]
        )
        self.proj = torch.nn.Linear(
            self.config.d_model, self.config.d_model, bias=False
        )
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.concat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FusedMultiHeadAttention(torch.nn.Module):
    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.attn = torch.nn.Linear(
            self.config.d_model, self.config.d_model * 3, bias=False
        )
        self.proj = torch.nn.Linear(
            self.config.d_model, self.config.d_model, bias=False
        )
        self.dropout = torch.nn.Dropout(self.config.dropout)

        self.q_rope = torchtune.modules.RotaryPositionalEmbeddings(
            self.config.d_model // self.config.n_heads, self.config.block_size
        )
        self.k_rope = torchtune.modules.RotaryPositionalEmbeddings(
            self.config.d_model // self.config.n_heads, self.config.block_size
        )

    def __reshape_view(self, w: torch.Tensor, B: int, T: int, C: int) -> torch.Tensor:
        # (B, T, C) -> (B, T, n_head, head_dim)
        return w.view(B, T, self.config.n_heads, C // self.config.n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = tuple(
            map(
                lambda w: self.__reshape_view(w, B, T, C),  # .transpose(1, 2),
                self.attn(x).split(self.config.d_model, dim=2),
            )
        )  # (B, T, n_head, head_dim)

        q: torch.Tensor = self.q_rope(q).transpose(1, 2)  # (B, n_head, T, head_dim)
        k: torch.Tensor = self.k_rope(k).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Flash
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v.transpose(1, 2),
            attn_mask=None,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=True,
        )

        out = self.dropout(self.proj(out.transpose(1, 2).contiguous().view(B, T, C)))
        return out


class SwiGLU(torch.nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()  # type: ignore
        self.w = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.silu(x) * self.w(x)
        return out


class FeedForward(torch.nn.Module):
    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.w1 = torch.nn.Linear(
            self.config.d_model, self.config.d_model * 4, bias=False
        )
        self.swish = torch.nn.SiLU()
        self.w2 = torch.nn.Linear(
            self.config.d_model, self.config.d_model * 4, bias=False
        )
        self.w3 = torch.nn.Linear(
            self.config.d_model * 4, self.config.d_model, bias=False
        )
        self.drop = torch.nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.w3(self.swish(self.w1(x)) * self.w2(x)))
        return out


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.sa = FusedMultiHeadAttention(self.config)
        self.ff = FeedForward(self.config)
        self.rmsn1 = torchtune.modules.RMSNorm(self.config.d_model)
        self.rmsn2 = torchtune.modules.RMSNorm(self.config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.rmsn1(x))
        x = x + self.ff(self.rmsn2(x))
        return x


class Transformer(torch.nn.Module):
    config = FernConfig

    def __init__(self, config: FernConfig):
        super().__init__()  # type: ignore
        self.config = config
        self.token_embedding_table = torch.nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )
        self.token_embedding_table.weight.data *= 0.1
        self.blocks = torch.nn.Sequential(
            *[TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.rmsn = torchtune.modules.RMSNorm(self.config.d_model)
        self.lm_head = torch.nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: t.Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, t.Optional[torch.Tensor]]:
        tok_emb: torch.Tensor = self.token_embedding_table(idx)  # (B, T, C)
        x = tok_emb
        x: torch.Tensor = self.blocks(x)
        x: torch.Tensor = self.rmsn(x)
        # logits: torch.Tensor = self.lm_head(x)  # (B, T, vocab_size)
        logits = x @ self.token_embedding_table.weight.transpose(-2, -1)

        loss = None
        if targets is not None:
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self,
        indexes: torch.Tensor,
        max_new_tokens: t.Optional[int] = None,
        stop_token: t.Optional[int] = None,
    ) -> t.Generator[torch.Tensor, None, None]:
        assert (
            max_new_tokens is not None or stop_token is not None
        ), "Either `max_new_tokens` or `stop_token` should be set"
        i = 0
        idx_next = torch.tensor([[-1]])
        while_conditions: list[t.Callable[[None], bool]] = []
        if max_new_tokens is not None:
            while_conditions.append(lambda _: i < max_new_tokens)
        if stop_token is not None:
            while_conditions.append(lambda _: stop_token not in idx_next)
        while all(map(lambda x: x(None), while_conditions)):
            i += 1
            # crop idx
            idx_cond = indexes[:, -self.config.block_size :]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            indexes = torch.cat((indexes, idx_next), dim=1)
            yield idx_next

    def get_generation(
        self,
        indexes: torch.Tensor,
        max_new_tokens: t.Optional[int] = None,
        stop_token: t.Optional[int] = None,
    ) -> torch.Tensor:
        result: list[int] = indexes[0].tolist()  # type: ignore
        for tok in self.generate(indexes, max_new_tokens, stop_token):
            result.append(int(tok.item()))
        return torch.tensor(result)
