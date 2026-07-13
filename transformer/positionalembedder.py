import torch

class PositionalEmbedder(torch.nn.Module):

    def __init__(self, maxLen: int, dModel: int) -> None:
        super().__init__()

        self.positionalEmbeddings = torch.nn.Embedding(maxLen, dModel)

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