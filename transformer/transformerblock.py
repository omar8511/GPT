import torch
from transformer.multiheadattention import MultiHeadAttention
from transformer.feedforward import FeedForward
from transformer.residuals import Residuals
from transformer.config import Config
from transformer.kvcache import KVCache

class TransformerBlock(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.residualsOne, self.residualsTwo = Residuals(config), Residuals(config)
        self.attention = MultiHeadAttention(config)
        self.feedForward = FeedForward(config)


    def forward(self, x: torch.Tensor, offset: int = 0,  cache: list[KVCache] = None) -> torch.Tensor:
        """
        Passes input through residuals, attention and feedforward layer
        
        Args:
        x - Input tensor of shape (..., dModel)
        offset - current index of the input
        cache - list of N KVCache instances

        Returns:
        layerTwo - Tensor of shape (...., dModel)
        
        """
        layerOne = self.residualsOne(x, lambda y: self.attention(y, offset, cache))
        layerTwo = self.residualsTwo(layerOne, self.feedForward)

        return layerTwo

