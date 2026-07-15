import torch
from transformer.positionalembedder import PositionalEmbedder

class Embedder(torch.nn.Module):

    def __init__(self, vocabSize: int) -> None:
        super().__init__()
        self.dModel = 512
        self.vocabSize = vocabSize
        self.positionalEncoder = PositionalEmbedder(256, self.dModel)

        self.embedding = torch.nn.Embedding(vocabSize + 1, self.dModel, padding_idx=vocabSize) # Matrix init to vocabSize x dModel 
        

    # when you do module(...) it implicitly calls forward for a tensor allows us to do self.embedder(input)
    def forward(self, inputs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps each token id to its learned embedding vector, padding sequences to equal length before lookup.

        Args:
        Inputs: list of tokens in their integer encoding 

        Returns:
        maskTensor (len(inputs), maxLength, (1))

        embeddings (len(inputs), maxLength, dModel)
        """
        copies = [row[:] for row in inputs]
        maxSeqLen = len(max(inputs, key=len))

        for copy in copies:
            while len(copy) < maxSeqLen:
                copy.append(self.vocabSize)


        copyTensor = torch.tensor(copies)
        maskTensor = copyTensor == self.vocabSize
        tokenEmbeddings = self.embedding(copyTensor)
        positionEmbeddings = self.positionalEncoder(maxSeqLen)


        # broadcasts over so each maxLength block gets the same positional encodings which makes sense
        return maskTensor, (tokenEmbeddings + positionEmbeddings)
