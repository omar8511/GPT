import torch 
import math

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, maskTensor: torch.Tensor):
        super().__init__()
        self.dModel = 512
        self.numHeads = 8
        self.maskTensor = self.maskTensor

        self.wQuery = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wKey = torch.nn.Linear(self.dModel, self.dModel, bias=False)
        self.wValue = torch.nn.Linear(self.dModel, self.dModel, bias=False)

        # Input shape is batch, maxLen, dModel so we multiply rows and get 1 * dModel
        

    def forward(self, x: torch.Tensor):
        Q, K, V = x @ self.wQuery, x @ self.wKey, x @ self.wValue

        # batch, maxlen, dModel
        dK = self.dModel / self.numHeads


        torch.reshape(Q, (batch, maxLen, self.numHeads, self.dModel))
        torch.reshape(K, (batch, maxLen, self.numHeads, self.dModel))
        torch.reshape(V, (batch, maxLen, self.numHeads, self.dModel))
        # Batch each row of each matrix into a numHeads x dK matrix 

        Q = torch.transpose(Q, 1, 2) 
        K = torch.transpose(K, 1, 2) 
        V = torch.transpose(V, 1, 2) 
        # batch, numHeads, maxLen, dK


        #
        #yes that makes sense and so now we do transposing what exactly happens there cuz it doenst sit in my head that the size of the list of matrices      
        #becomes its dimension and the dim becomes the size of the list is it that you group each row together into a new matrix basically and thats what     
        #swaps the dims   


        K = torch.transpose(K, 2, 3) 
        # K is batch, numHeads, dK, maxLen

        self.maskTensor = torch.reshape(self.maskTensor, (batch, 1, 1, maxLen))
        A = Q @ K
        AScaled = A / math.sqrt(dK)
        # A is batch, numHeads, maxLen, maxLen, scaled dot product

        AMasked = torch.masked_fill(AScaled, self.maskTensor, -math.inf)

        weights = torch.softmax(AMasked, dim=-1)

        # Each Row sums to one

        weightedSum = weights @ V
        # (batch, numHeads, maxLen, dK)

        weightedSum = weightedSum.transpose(2, 3)

        