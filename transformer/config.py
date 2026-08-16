from dataclasses import dataclass

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
    numEpochs: int = 25
    batchSize: int = 25
    seed: int = 1234
    valFraction: float = 0.1 # Split train / validation
    evalEvery: int = 20 # How often to run the validaiton pass
    maxLr: float = 1e-3
    minLr: float = 1e-4
    warmupSteps: int = 100
    weightDecay: float = 0.1