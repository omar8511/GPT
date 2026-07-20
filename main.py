from transformer.generate import generate
from transformer.train import train
from transformer.config import Config
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt")
    parser.add_argument("trainFile", type=str, help="Path to training corpus")
    parser.add_argument("--maxTokens", type=int, default=50)
    args = parser.parse_args()

    config = Config()
    gpt, tokeniser = train(args.trainFile, config)
    print(generate(gpt, tokeniser, args.prompt, args.maxTokens, config))
    

main()


