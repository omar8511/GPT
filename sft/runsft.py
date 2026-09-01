from transformer.config import Config
from sft.sft import sft

def runsft():
    c = Config()
    c.batchSize = 8
    c.numEpochs = 4
    c.useLoRA = True

    if c.useLoRA:
        c.maxLr = 1e-4         
        c.minLr = 1e-5
    else:
        c.maxLr = 1e-5        # ~60x below pretraining — SFT nudges, it doesn't reshape
        c.minLr = 1e-6
    
    c.warmupSteps = 5
    c.logEvery = 2
    c.evalEvery = 10
    c.valFraction = 0.15    

    out = 'trained_model/sft_lora.pt' if c.useLoRA else 'trained_model/sft.pt'
    sft('trained_model/checkpoint.pt', 'sft/examples.jsonl', out, c)

if __name__ == "__main__":
      runsft()