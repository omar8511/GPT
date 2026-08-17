import torch
from transformer.positionalembedder import PositionalEmbedder
from transformer.config import Config

class Embedder(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dModel = config.dModel
        self.vocabSize = config.vocabSize
        self.positionalEncoder = PositionalEmbedder(config)

        self.embedding = torch.nn.Embedding(self.vocabSize + 1, self.dModel, padding_idx=self.vocabSize) # Matrix init to vocabSize x dModel 
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.embedding.weight.data[self.vocabSize].zero_()
        self.dropout = torch.nn.Dropout(config.dropout)
        

    # when you do module(...) it implicitly calls forward for a tensor allows us to do self.embedder(input)
    def forward(self, inputs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps each token id to its learned embedding vector, padding sequences to equal length before lookup.

        Args:
        Inputs: list of tokens in their integer encoding 

        Returns:
        maskTensor (len(inputs), maxLength)

        embeddings (len(inputs), maxLength, dModel)
        """
        copies = [row[:] for row in inputs]
        maxSeqLen = len(max(inputs, key=len))

        copyTensor = torch.tensor(copies)
        maskTensor = copyTensor == self.vocabSize
        tokenEmbeddings = self.embedding(copyTensor)
        positionEmbeddings = self.positionalEncoder(maxSeqLen)


        # broadcasts over so each maxLength block gets the same positional encodings which makes sense
        return maskTensor, self.dropout(tokenEmbeddings + positionEmbeddings)
