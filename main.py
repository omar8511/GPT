from transformer.generate import generate
from transformer.train import train
from transformer.config import Config
from transformer.checkpoint import loadCheckpoint
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt")
    parser.add_argument("model", type=str, help="Path to model")
    parser.add_argument("--maxTokens", type=int, default=50)
    parser.add_argument("--trainFile", type=str, help="Path to training corpus", default=None)
    args = parser.parse_args()

    
    if os.path.exists(args.model):
      gpt, tokeniser, config = loadCheckpoint(args.model)
    else:
      config = Config()
      gpt, tokeniser = train(args.trainFile, config, args.model)

    print(generate(gpt, tokeniser, args.prompt, args.maxTokens, config))
    

main()


