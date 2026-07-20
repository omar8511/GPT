from transformer.generate import generate
from transformer.train import train
from transformer.config import Config
from transformer.checkpoint import Checkpoint
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt")
    parser.add_argument("trainFile", type=str, help="Path to training corpus")
    parser.add_argument("--maxTokens", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    
    if os.path.exists(args.checkpoint):
      gpt, tokeniser, config = Checkpoint.loadCheckpoint(args.checkpoint)
    else:
      config = Config()
      gpt, tokeniser = train(args.trainFile, config, args.checkpoint)

    print(generate(gpt, tokeniser, args.prompt, args.maxTokens, config))
    

main()


