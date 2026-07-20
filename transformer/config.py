from dataclasses import dataclass

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