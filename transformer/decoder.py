import torch
from transformer.config import Config
from transformer.transformerblock import TransformerBlock
from transformer.kvcache import KVCache

class Decoder(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.N = config.N
        self.blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(self.N)])


    def forward(self, x: torch.Tensor, offset: int = 0, cache: KVCache = None) -> torch.Tensor:
        """
        Passes x through N layers

        Args:
        x - Input tensor of shape (..., dModel)
        offset - current index of the input
        cache - KVCache instance

        Returns
        x - Input tensor after being passed through N layers shape (..., dModel)
        """
        for i, block in enumerate(self.blocks):
            x = block(x, offset, cache[i] if cache is not None else None)

        return x