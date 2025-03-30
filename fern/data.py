import torch
from tqdm.auto import tqdm

from fern.config import FernConfig


def sample_batch(
    data: torch.Tensor, fern_config: FernConfig, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - fern_config.block_size, (batch_size,))
    x = torch.stack([data[i : i + fern_config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + fern_config.block_size + 1] for i in ix])
    return x, y


@torch.no_grad()  # type: ignore
def estimate_loss(
    m: torch.nn.Module,
    data: torch.Tensor,
    fern_config: FernConfig,
    batch_size: int,
    eval_iters: int,
) -> torch.Tensor:
    m.eval()
    losses = torch.empty(eval_iters)
    for i in tqdm(range(eval_iters)):
        bX, bY = sample_batch(data, fern_config, batch_size)
        _logits, loss = m(bX, bY)
        losses[i] = loss
    m.train()
    return losses.mean()
