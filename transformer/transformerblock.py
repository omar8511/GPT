import torch
from transformer.multiheadattention import MultiHeadAttention
from transformer.feedforward import FeedForward
from transformer.residuals import Residuals
from transformer.config import Config

class TransformerBlock(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.residualsOne, self.residualsTwo = Residuals(config), Residuals(config)
        self.attention = MultiHeadAttention(config)
        self.feedForward = FeedForward(config)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through residuals, attention and feedforward layer

        Args:
        x - Input tensor of shape (..., dModel)

        Returns:
        layerTwo - Tensor of shape (...., dModel)
        
        """
        layerOne = self.residualsOne(x, lambda y: self.attention(y))
        layerTwo = self.residualsTwo(layerOne, self.feedForward)

        return layerTwo

