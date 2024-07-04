import typing as t

import torch
import torchtune  # type: ignore

import tqdm

from fern.config import FernConfig


batch_size = 32  # 64
max_iters = 10000  # 5000
eval_interval = 1000  # 500
learning_rate = 3e-4
device = "cuda"
eval_iters = 100  # 200

torch.manual_seed(0)  # type: ignore
torch.set_float32_matmul_precision("high")

with open("data/books_concat.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = dict(enumerate(chars))
stoi = dict((s, i) for i, s in itos.items())

encode: t.Callable[[str], list[int]] = lambda seq: [stoi[ch] for ch in seq]
decode: t.Callable[[list[int]], str] = lambda seq: "".join(itos[ix] for ix in seq)

data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.8 * len(data))

train_data = data[:n]
val_data = data[n:]

fern_config = FernConfig(
    d_model=384,
    n_heads=6,
    n_layers=6,
    vocab_size=vocab_size,
    block_size=512,  # 256
    dropout=0.2,
)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - fern_config.block_size, (batch_size,))
    x = torch.stack([data[i : i + fern_config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + fern_config.block_size + 1] for i in ix])
    return x, y


@torch.no_grad()  # type: ignore
def estimate_loss(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.empty(eval_iters)
        for i in range(eval_iters):
            bX, bY = get_batch(split)
            _logits, loss = m(bX, bY)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        # (B, T, C) -> (B, T, n_head, head_dim)
        reshape_view: t.Callable[[torch.Tensor], torch.Tensor] = lambda w: w.view(
            B, T, self.config.n_heads, C // self.config.n_heads
        )
        q, k, v = tuple(
            map(
                reshape_view,  # .transpose(1, 2),
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
        # head_size = self.config.d_model // self.config.n_heads
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
        logits: torch.Tensor = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(
        self, indexes: torch.Tensor, max_new_tokens: int
    ) -> t.Generator[torch.Tensor, None, None]:
        for _ in range(max_new_tokens):
            # crop idx
            idx_cond = indexes[:, -self.config.block_size :]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            indexes = torch.cat((indexes, idx_next), dim=1)
            yield idx_next

    def get_generation(
        self, indexes: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        result = torch.empty(
            (indexes.size(0), indexes.size(1) + max_new_tokens),
            dtype=indexes.dtype,
            device=indexes.device,
        )
        result[0, 0:-max_new_tokens] = indexes
        current_token = 0
        for tok in self.generate(indexes, max_new_tokens):
            result[0, current_token - max_new_tokens] = tok
            current_token += 1
        return result


if __name__ == "__main__":
    model = Transformer(fern_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in tqdm.tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            estimated_loss = estimate_loss(model)
            print(f"Estimated loss at iteration {iter}: {estimated_loss}")
            torch.save(  # type: ignore
                {
                    "epoch": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": estimated_loss,
                },
                f"model-{iter}.pt",
            )
        x, y = get_batch("train")
        _logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    estimated_loss = estimate_loss(model)
    print(f"Estimated loss at iteration {max_iters}: {estimated_loss}")
    torch.save(  # type: ignore
        {
            "epoch": max_iters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": estimated_loss,
        },
        f"model-{max_iters}.pt",
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    generated: list[int] = model.get_generation(context, 512)[0].tolist()  # type: ignore
    print(decode(generated))
