import torch

class LayerNorm(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dModel = 512
        self.gamma = torch.nn.Parameter(torch.ones((self.dModel)))
        self.beta = torch.nn.Parameter(torch.zeros((self.dModel)))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns LN(x) for an input

        Args:
        x - Tensor of shape (..., dModel)

        Returns
        output - Tensor of shape (..., dModel) with LN applied
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        eps = 1e-5
        normalised = (x - mean) / torch.sqrt(var + eps)

        output = self.gamma * normalised + self.beta
        return output