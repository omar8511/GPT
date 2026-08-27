from dataclasses import dataclass
import torch

@dataclass
class Config:
    dModel: int = 512
    dFF: int = 2048
    numHeads: int = 8
    N: int = 4 
    maxLen: int = 256
    vocabSize: int = None
    temperature: float =1.0
    k: int = 10
    numMerges: int = 500
    totalSteps: int = 1700 # No more numEpochs - random sampling from a memmap has no notion of "one pass"
    numValBatches: int = 10 # Fixed number of val batches, sampled once and reused every eval
    batchSize: int = 25
    seed: int = 1234
    valFraction: float = 0.1 # Split train / validation
    evalEvery: int = 20 # How often to run the validaiton pass
    maxLr: float = 1e-3
    minLr: float = 1e-4
    warmupSteps: int = 100
    weightDecay: float = 0.1
    dropout: float = 0.1
    isDebug: bool = False
    logEvery: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"