from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from tokeniser.tokeniserutils import streamFileSymbols, streamTrainingSequences, batchSequences
from transformer.gpt import GPT
from transformer.transfomerutils import buildTrainingPairs
from tokeniser.tokeniser import Tokeniser
from transformer.config import Config


import torch


def train(filePath: str, config: Config) -> tuple[GPT, Tokeniser]:

    preprocesser = PreProcessor()
    stream = streamFileSymbols(filePath, preprocesser)
    trainer = Trainer(stream, config)
    merges, tokens = trainer.train_BPE()
    tokens.update(preprocesser.byteEncoder.values())
    bpe = BPE(merges, tokens)
    config.vocabSize = len(tokens)    

    gpt = GPT(config)
    
    optimiser = torch.optim.AdamW(gpt.parameters())

    for _ in range(config.numEpochs):
        trainingStream = streamTrainingSequences(filePath, preprocesser, bpe, config.maxLen)
        for batch in batchSequences(trainingStream, config.batchSize):
            input, target = buildTrainingPairs(batch)
            maxLen = len(max(input, key=len))
            for t in target:
                while len(t) < maxLen:
                    t.append(config.vocabSize)
            targetTensor = torch.tensor(target)

            logits = gpt(input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocabSize), targetTensor.reshape(-1), ignore_index=config.vocabSize)
            print(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    tokeniser = Tokeniser(preprocesser, bpe)


    return gpt, tokeniser

