import numpy as np
import torch


class MemmapSampler:
    """
    Samples random maxLen+1 token windows from a flat uint16 array on disk
    (written by tokeniser.tokeniserutils.tokeniseToDisk), without loading
    the whole file into memory.
    """

    def __init__(self, path: str, maxLen: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.maxLen = maxLen

    def __len__(self) -> int:
        return len(self.data)

    def sampleBatch(self, batchSize: int) -> torch.Tensor:
        """
        Args:
        batchSize: Number of windows to sample

        Returns:
        (batchSize, maxLen + 1) LongTensor of token ids, at random offsets
        into the file. Offsets are independent and drawn with replacement,
        so windows may overlap across a batch or across calls.
        """
        highestOffset = len(self.data) - self.maxLen - 1
        offsets = torch.randint(0, highestOffset, (batchSize,))
        windows = np.stack([self.data[o : o + self.maxLen + 1] for o in offsets.tolist()])
        return torch.from_numpy(windows.astype(np.int64))
