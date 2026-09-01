import torch
from transformer.config import Config

class LoRALinear(torch.nn.Module):

    def __init__(self, linear: torch.nn.Linear, config: Config):
        super().__init__()
        self.base = linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # A is random and B is zero so BA is zero at init: the wrapped layer starts
        # out identical to the base. Both zero would leave both gradients zero forever.
        self.A = torch.nn.Parameter(torch.empty((config.loraRank, linear.in_features)).normal_(mean=0, std=0.02))
        self.B = torch.nn.Parameter(torch.zeros((linear.out_features, config.loraRank)))
        self.scale = config.loraAlpha / config.loraRank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the frozen layer and adds the low rank correction

        Args:
        x: Tensor of shape (..., inFeatures)

        Returns:
        Tensor of shape (..., outFeatures)
        """
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
