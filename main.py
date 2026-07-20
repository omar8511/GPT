from tokeniser.tokeniser import Tokeniser
from transformer.gpt import GPT
from transformer.generate import generate
from transformer.train import train
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt")
    parser.add_argument("trainFile", type=str, help="Path to training corpus")
    parser.add_argument("--maxTokens", type=int, default=50)
    args = parser.parse_args()

    gpt, tokeniser = train(args.trainFile)
    print(generate(gpt, tokeniser, args.prompt, args.maxTokens))
    

main()


