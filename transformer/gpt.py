import torch
from transformer.embedder import Embedder
from transformer.decoder import Decoder
from transformer.config import Config
from transformer.layernorm import LayerNorm
from transformer.kvcache import KVCache

class GPT(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.vocabSize = config.vocabSize
        self.embedder = Embedder(config)
        self.decoder = Decoder(config)
        self.LN = LayerNorm(config)
        self.LMBias = torch.nn.Parameter(torch.zeros(self.vocabSize))

    def forward(self, inputs: torch.Tensor, cache: list[KVCache] = None):
        """
        Returns the model output given a list of words as their integer encodings from BPE

        Args:
        inputs - List of words as integer encodings from BPE
        cache - list of N KVCache instances

        Returns:
        result - Tensor of size (batch, maxSeqLen, vocabSize) 
        """
        offset = cache[0].length() if cache is not None else 0

        embeddings = self.embedder(inputs, offset)
        decodings = self.decoder(embeddings, offset, cache)
        output = self.LN(decodings)
        result = torch.nn.functional.linear(output, self.embedder.embedding.weight[:self.vocabSize], self.LMBias)
        return result
    
