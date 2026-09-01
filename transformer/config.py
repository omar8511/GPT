from dataclasses import dataclass
import torch

@dataclass
class Config:
    dModel: int = 512
    dFF: int = 2048
    numHeads: int = 8
    N: int = 8
    maxLen: int = 256
    vocabSize: int = None
    temperature: float =1.0
    k: int = 10
    numMerges: int = 8000
    totalSteps: int = 36000
    numEpochs: int = 3 # SFT is epoch-based over a small example set, unlike pretraining
    numValBatches: int = 10
    batchSize: int = 64
    seed: int = 1234
    valFraction: float = 0.1 # Train/val split
    evalEvery: int = 250 
    maxLr: float = 6e-4
    minLr: float = 6e-5
    warmupSteps: int = 500 
    weightDecay: float = 0.1
    dropout: float = 0.1
    isDebug: bool = False
    logEvery: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bpeTrainLines: int = 300000
    useAmp: bool = torch.cuda.is_available() # bf16 autocast - CUDA only, CPU stays fp32
    checkpointEvery: int = 1000 # Overwrites one file, so an interrupted run loses at most this many steps
    useLoRA: bool = False
    loraRank: int = 8
    loraAlpha: int = 16