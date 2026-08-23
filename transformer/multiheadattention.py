import torch 
import math
from transformer.config import Config


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

        self.causalMask = torch.register_buffer(
            "causalMask",
            torch.triu(torch.ones(config.maxLen, config.maxLen, dtype=torch.bool), diagonal=1),
            persistent=False # Wont be saved as a parameter
        )

        # Input shape is batch, maxLen, dModel so we multiply rows and get 1 * dModel
        

    def forward(self, x: torch.Tensor, maskTensor: torch.Tensor):
        """
        Performs the Multi Head Attention's steps

        Args:
        x - Tensor of shape (batch, maxLen(acrossBatch), dModel)
        maskTensor - (batch, maxLen), True at positions that are padding and false at real tokens

        Returns:
        output - Tensor of shape (batch, maxLen(acrossBatch), dModel)
        """
        Q, K, V = self.wQuery(x), self.wKey(x), self.wValue(x)
        (batch, maxLen, _) = x.shape
        causalMask = self.causalMask[:maxLen, :maxLen]

        # batch, maxlen, dModel
        dK = self.dModel // self.numHeads


        Q = torch.reshape(Q, (batch, maxLen, self.numHeads, dK))
        K = torch.reshape(K, (batch, maxLen, self.numHeads, dK))
        V = torch.reshape(V, (batch, maxLen, self.numHeads, dK))
        # Batch each row of each matrix into a numHeads x dK matrix 

        Q = torch.transpose(Q, 1, 2) 
        K = torch.transpose(K, 1, 2) 
        V = torch.transpose(V, 1, 2) 
        # batch, numHeads, maxLen, dK


        K = torch.transpose(K, 2, 3) 
        # K is batch, numHeads, dK, maxLen

        mask = torch.reshape(maskTensor, (batch, 1, 1, maxLen)) | causalMask
        A = Q @ K
        AScaled = A / math.sqrt(dK)
        # A is batch, numHeads, maxLen, maxLen, scaled dot product

        AMasked = torch.masked_fill(AScaled, mask, -math.inf)

        weights = torch.softmax(AMasked, dim=-1)
        # Each Row sums to one
        if self.isDebug:
            self.attentionWeights = weights.detach()

        weights = self.dropout(weights)

        weightedSum = weights @ V
        # (batch, numHeads, maxLen, dK)

        weightedSum = weightedSum.transpose(1, 2)
        # (batch, maxLen, numHeads, dK)

        weightedSum = torch.reshape(weightedSum, (batch, maxLen, self.dModel))

        output = self.wOut(weightedSum)

        # (batch, maxLen, dModel)
        return output

