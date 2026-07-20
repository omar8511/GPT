import torch
from transformer.embedder import Embedder
from transformer.decoder import Decoder
from transformer.config import Config
from transformer.layernorm import LayerNorm

class GPT(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.vocabSize = config.vocabSize
        self.embedder = Embedder(config)
        self.decoder = Decoder(config)
        self.LN = LayerNorm(config)
        self.LMBias = torch.nn.Parameter(torch.zeros(self.vocabSize))

    def forward(self, inputs: list[list[int]]):
        """
        Returns the model output given a list of words as their integer encodings from BPE

        Args:
        Inputs - List of words as integer encodings from BPE

        Returns:
        result - Tensor of size (batch, maxSeqLen, vocabSize) 
        """
        mask, embeddings = self.embedder(inputs)
        decodings = self.decoder(embeddings, mask)
        output = self.LN(decodings)
        result = torch.nn.functional.linear(output, self.embedder.embedding.weight[:self.vocabSize], self.LMBias)
        return result
    
