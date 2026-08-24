import torch
from transformer.config import Config
from transformer.transformerblock import TransformerBlock

class Decoder(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.N = config.N
        self.blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(self.N)])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes x through N layers

        Args:
        x - Input tensor of shape (..., dModel)

        Returns
        x - Input tensor after being passed through N layers shape (..., dModel)
        """
        for block in self.blocks:
            x = block(x)

        return x