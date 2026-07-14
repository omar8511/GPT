import torch 
import math

class MultiHeadAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dModel = 512
        self.numHeads = 8

        self.wQuery = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wKey = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wValue = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wOut = torch.nn.Linear(self.dModel, self.dModel, bias=False)

        # Input shape is batch, maxLen, dModel so we multiply rows and get 1 * dModel
        

    def forward(self, x: torch.Tensor, maskTensor: torch.Tensor):
        """
        Performs the Multi Head Attention's steps

        Args:
        x - Tensor of shape (batch, maxLen(acrossBatch), dModel)

        Returns:
        output - Tensor of shape (batch, maxLen(acrossBatch), dModel)
        """
        Q, K, V = self.wQuery(x), self.wKey(x), self.wValue(x)
        (batch, maxLen, _) = x.shape
        causalMask = torch.triu(torch.ones(maxLen, maxLen, dtype=torch.bool), diagonal=1)

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

        weightedSum = weights @ V
        # (batch, numHeads, maxLen, dK)

        weightedSum = weightedSum.transpose(1, 2)
        # (batch, maxLen, numHeads, dK)

        weightedSum = torch.reshape(weightedSum, (batch, maxLen, self.dModel))

        output = self.wOut(weightedSum)

        # (batch, maxLen, dModel)
        return output

