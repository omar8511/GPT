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

        # Input shape is batch, maxLen, dModel so we multiply rows and get 1 * dModel
        

    def forward(self):
        Q, K, V = x @ self.wQuery, x @ self.wKey, x @ self.wValue

        # batch, maxlen, dModel
        dK = self.dModel / self.numHeads


        torch.reshape(Q, (batch, maxLen, self.numHeads, self.dModel))
        torch.reshape(K, (batch, maxLen, self.numHeads, self.dModel))
        torch.reshape(V, (batch, maxLen, self.numHeads, self.dModel))
        # Batch each row of each matrix into a numHeads x dModel matrix 

        torch.transpose(Q, 1, 2) 
        torch.transpose(K, 1, 2) 
        torch.transpose(V, 1, 2) 
        # batch, numHeads, maxLen, dK


        torch.transpose(K, 2, 3) 
        # K is batch, numHeads, dK, maxLen

        A = Q @ K
        AScaled = A / math.sqrt(dK)
        # A is batch, numHeads, maxLen, maxLen, scaled dot product

