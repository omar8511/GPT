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
    def forward(self, inputs: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Maps each token id to its learned embedding vector,

        Args:
        Inputs: (batchSize, maxLen)
        offset: current index to embed from

        Returns:

        embeddings (len(inputs), maxLen, dModel)
        """
        
        tokenEmbeddings = self.embedding(inputs)
        positionEmbeddings = self.positionalEncoder(inputs.shape[1], offset)


        # broadcasts over so each maxLength block gets the same positional encodings which makes sense
        return self.dropout(tokenEmbeddings + positionEmbeddings)
