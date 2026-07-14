import torch 


class FeedForward(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dModel = 512
        self.dFF = 2048
        self.linearOne = torch.nn.Linear(self.dModel, self.dFF)
        self.linearTwo = torch.nn.Linear(self.dFF, self.dModel)
        self.GELU = torch.nn.GELU()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU to an input and scales it to dFF and back

        Args:
        x - Tensor of shape (..., dModel)

        returns 
        xTwo - Tensor of shape (..., dModel) with GELU applied

        
        """
        xOne = self.linearOne(x)
        xGELU = self.GELU(xOne)
        xTwo = self.linearTwo(xGELU)
        return xTwo
        
