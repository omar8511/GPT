from transformer.config import Config
from sft.sft import sft

def runsft():
    c = Config()
    c.batchSize = 8
    c.numEpochs = 4
    c.maxLr = 1e-5          # ~60x below pretraining — SFT nudges, it doesn't reshape
    c.minLr = 1e-6
    c.warmupSteps = 5
    c.logEvery = 2
    c.evalEvery = 10
    c.valFraction = 0.15    

    sft('trained_model/checkpoint.pt', 'sft/examples.jsonl', 'trained_model/sft.pt', c)

if __name__ == "__main__":
      runsft()