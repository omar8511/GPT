import torch
from transformer.config import Config


class PositionalEmbedder(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.positionalEmbeddings = torch.nn.Embedding(config.maxLen, config.dModel)

    def forward(self, seqLength: int) -> torch.Tensor:
        """
        Returns a (seqLen, dModel) tensor of positional embeddings

        Args:
        seqLength: range that you need positional embeddings for

        Returns:
        (seqLen, dModel) tensor of positional encodings
        """
        positions = torch.arange(seqLength)
        return self.positionalEmbeddings(positions)