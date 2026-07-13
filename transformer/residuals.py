import torch
from transformer.layernorm import LayerNorm
from typing import Callable


class Residuals(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layerNorm = LayerNorm()


    def forward(self, x: torch.Tensor, subLayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Returns x + subLayer(lN(x))

        Args:
        x - tensor of shape (..., dModel)
        subLayer - subLayer to this residual

        Returns
        output - tensor of shape (..., dModel)
        """
        output = x + subLayer(self.layerNorm(x))
        return output
    
