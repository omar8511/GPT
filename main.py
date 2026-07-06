from tokeniser.preprocessor import PreProcessor
from tokeniser.trainer import Trainer
from tokeniser.bpe import BPE
from tokeniser.tokeniserutils import streamFileSymbols
from tokeniser.tokeniser import Tokeniser

def main():
    preProcessor = PreProcessor()
    stream = streamFileSymbols("tokeniser/test.txt", preProcessor)
    trainer = Trainer(stream)
    merges, tokens = trainer.train_BPE(5)
    tokens.update(preProcessor.byteEncoder.values())
    bpe = BPE(merges, tokens)
    tokeniser = Tokeniser(preProcessor, bpe)

    encodings = tokeniser.encode("Hi ⚽️🎮👩🏿‍🎨")
    decodings = tokeniser.decode(encodings)

    print(decodings)
    print("Hi ⚽️🎮👩🏿‍🎨")
    


main()