import torch 
import math
from transformer.config import Config
from transformer.kvcache import KVCache


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.dModel = config.dModel
        self.numHeads = config.numHeads

        self.wQuery = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wKey = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wValue = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wOut = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.isDebug = config.isDebug
        self.attentionWeights = None

        self.register_buffer(
            "causalMask",
            torch.triu(torch.ones(config.maxLen, config.maxLen, dtype=torch.bool), diagonal=1),
            persistent=False # Wont be saved as a parameter
        )

        # Input shape is batch, maxLen, dModel so we multiply rows and get 1 * dModel
        

    def forward(self, x: torch.Tensor, offset: int = 0, cache: KVCache = None):
        """
        Performs the Multi Head Attention's steps

        Args:
        x - Tensor of shape (batch, maxLen(acrossBatch), dModel)
        offset - current index of the input
        cache - KVCache instance

        Returns:
        output - Tensor of shape (batch, maxLen(acrossBatch), dModel)
        """
        Q, K, V = self.wQuery(x), self.wKey(x), self.wValue(x)
        (batch, qLen, _) = x.shape
        causalMask = self.causalMask[offset : offset + qLen, :offset + qLen]

        # batch, qLen, dModel
        dK = self.dModel // self.numHeads


        Q = torch.reshape(Q, (batch, qLen, self.numHeads, dK))
        K = torch.reshape(K, (batch, qLen, self.numHeads, dK))
        V = torch.reshape(V, (batch, qLen, self.numHeads, dK))
        # Batch each row of each matrix into a numHeads x dK matrix 

        Q = torch.transpose(Q, 1, 2) 
        K = torch.transpose(K, 1, 2) 
        V = torch.transpose(V, 1, 2) 
        # batch, numHeads, qLen, dK

        if cache is not None:
            K, V = cache.append(K, V)

        K = torch.transpose(K, 2, 3) 
        # K is batch, numHeads, dK, qLen

        A = Q @ K
        AScaled = A / math.sqrt(dK)
        # A is batch, numHeads, qLen, qLen, scaled dot product

        AMasked = torch.masked_fill(AScaled, causalMask, torch.finfo(AScaled.dtype).min)

        weights = torch.softmax(AMasked, dim=-1)
        # Each Row sums to one
        if self.isDebug:
            self.attentionWeights = weights.detach()

        weights = self.dropout(weights)

        weightedSum = weights @ V
        # (batch, numHeads, qLen, dK)

        weightedSum = weightedSum.transpose(1, 2)
        # (batch, qLen, numHeads, dK)

        weightedSum = torch.reshape(weightedSum, (batch, qLen, self.dModel))

        output = self.wOut(weightedSum)

        # (batch, qLen, dModel)
        return output

