import torch

class Embedder(torch.nn.Module):

    def __init__(self, vocabSize: int) -> None:
        super().__init__()
        self.dModel = 512
        self.vocabSize = vocabSize
        self.embedding = torch.nn.Embedding(vocabSize + 1, self.dModel, padding_idx=vocabSize) # Matrix init to vocabSize x dModel 
        

    # when you do module(...) it implicitly calls forward for a tensor allows us to do self.embedder(input)
    def forward(self, inputs: list[list[int]]) -> torch.Tensor:
        """
        Maps each token id to its learned embedding vector, padding sequences to equal length before lookup.

        Args:
        Inputs: list of tokens in their integer encoding 

        Returns:
        maskTensor (len(inputs), maxLength, (1))

        embeddings (len(inputs), maxLength, dModel)
        """
        copies = [row[:] for row in inputs]
        maxLength = len(max(inputs, key=len))

        for copy in copies:
            while len(copy) < maxLength:
                copy.append(self.vocabSize)


        copyTensor = torch.tensor(copies)
        maskTensor = copyTensor == self.vocabSize


            
        return maskTensor, self.embedding(torch.tensor(copies))
