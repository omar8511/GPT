import torch

class KVCache():

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates the new k, v tensors to the cache
        
        Args:
        k - key tensor to be appended to cache (batch, numHeads, qLen, dK)
        v - value tensor to be appended to cache (batch, numHeads, qLen, dK)

        Returns:
        (self.k, self.v) - cache with k, v appended
        """
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)

        return self.k, self.v


    def length(self) -> int:
        """
        Returns current offset 
        """
        return 0 if self.k is None else self.k.shape[2]

