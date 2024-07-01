from dataclasses import dataclass


@dataclass
class FernConfig:
    d_model: int  # Token embedding vector dimensionality
    n_heads: int  # Number of attention heads
    n_layers: int  # Number of Transformer Blocks
    vocab_size: int  # Number of tokens in vocabulary
    head_size: int  # Dimensionality of single attention head
    dropout: float  # Probability of dropout
    block_size: int  # Context length

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
        block_size: int,
    ):
        assert 0 < dropout < 1, "Dropout probability must be between (0, 1)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.head_size = self.d_model // self.n_heads
        self.dropout = dropout
        self.block_size = block_size
