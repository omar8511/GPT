import torch
from transformer.layernorm import LayerNorm
from typing import Callable
from transformer.config import Config




class Residuals(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.layerNorm = LayerNorm(config)


    def forward(self, x: torch.Tensor, subLayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Returns x + subLayer(lN(x))

        Args:
        x - tensor of shape (..., dModel)
        subLayer (torch.Tensor -> torch.Tensor) - subLayer to this residual

        Returns
        output - tensor of shape (..., dModel)
        """
        output = x + subLayer(self.layerNorm(x))
        return output
    
