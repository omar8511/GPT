import torch

from transformer.transformerblock import TransformerBlock

class Decoder(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.N = 4 
        self.blocks = torch.nn.ModuleList([TransformerBlock() for _ in range(self.N)])


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Passes x through N layers

        Args:
        x - Input tensor of shape (..., dModel)
        mask - (batch, maxLen), True at positions that are padding and false at real tokens

        Returns
        x - Input tensor after being passed through N layers shape (..., dModel)
        """
        for block in self.blocks:
            x = block(x, mask)

        return x