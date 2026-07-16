from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from tokeniser.tokeniserutils import streamFileSymbols, streamTrainingSequences, batchSequences
from transformer.gpt import GPT
from transformer.transfomerutils import buildTrainingPairs

import torch


def train() -> None:

    preprocesser = PreProcessor()
    stream = streamFileSymbols("tokeniser/test.txt", preprocesser)
    trainer = Trainer(stream)
    merges, tokens = trainer.train_BPE(25)
    tokens.update(preprocesser.byteEncoder.values())
    bpe = BPE(merges, tokens)
    vocabSize = len(tokens)
    

    gpt = GPT(512, vocabSize)
    
    optimiser = torch.optim.AdamW(gpt.parameters())
    numEpochs = 25

    for _ in range(numEpochs):
        trainingStream = streamTrainingSequences("traintext", preprocesser, bpe, 256)
        for batch in batchSequences(trainingStream, 25):
            input, target = buildTrainingPairs(batch)
            maxLen = len(max(input, key=len))
            for t in target:
                while len(t) < maxLen:
                    t.append(vocabSize)
            targetTensor = torch.Tensor(target)

            logits = gpt(input)
            loss = torch.F.cross_entropy(logits.reshape(-1, vocabSize), targetTensor.reshape(-1), ignore_index=vocabSize)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()




    filePath = ""


