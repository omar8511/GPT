from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from tokeniser.tokeniserutils import streamFileSymbols, streamTrainingSequences, batchSequences, countNonEmptyLines
from transformer.gpt import GPT
from transformer.transfomerutils import buildTrainingPairs
from tokeniser.tokeniser import Tokeniser
from transformer.config import Config
from transformer.checkpoint import saveCheckpoint
from traininglogger.logger import Logger
from itertools import islice


import torch


def train(filePath: str, config: Config, trainingPath: str) -> tuple[GPT, Tokeniser]:

    preprocesser = PreProcessor()
    stream = streamFileSymbols(filePath, preprocesser)
    trainer = Trainer(stream, config)
    merges, tokens = trainer.train_BPE()
    tokens.update(preprocesser.byteEncoder.values())
    bpe = BPE(merges, tokens)
    config.vocabSize = len(tokens)    
    logger = Logger(config)
    gpt = GPT(config)
    optimiser = torch.optim.AdamW(gpt.parameters())
    splitIndex = int(countNonEmptyLines(filePath) * (1 - config.valFraction))
    validationSymbolStream = islice(streamFileSymbols(filePath, preprocesser), splitIndex, None)
    valBatches = list(batchSequences(streamTrainingSequences(validationSymbolStream, bpe, config.maxLen), config.batchSize))

    step = 0
    for i in range(config.numEpochs):
        trainingSymbolStream = islice(streamFileSymbols(filePath, preprocesser), 0, splitIndex)
        trainingStream = streamTrainingSequences(trainingSymbolStream, bpe, config.maxLen)

        for batch in batchSequences(trainingStream, config.batchSize):
            inputSeqs, target = buildTrainingPairs(batch)
            targetTensor = torch.tensor(target)
            logits = gpt(inputSeqs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocabSize), targetTensor.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            logger.appendLoss(i, step, loss.item())
            step += 1

            if step % config.evalEvery == 0:
                gpt.eval()
                with torch.no_grad():
                    valTotalLoss = 0
                    for valBatch in valBatches:
                        inputSeqv, targetv = buildTrainingPairs(valBatch)
                        targetTensorv = torch.tensor(targetv)
                        logitsv = gpt(inputSeqv)
                        valLoss = torch.nn.functional.cross_entropy(logitsv.reshape(-1, config.vocabSize), targetTensorv.reshape(-1))
                        valTotalLoss += valLoss.item()
                    logger.appendLoss(i, step, loss.item(), valTotalLoss / len(valBatches))

                gpt.train()


    attentions = [block.attention.attentionWeights for block in gpt.decoder.blocks]
    attentionFirstBatch = attentions[0][0]
    firstBatch = inputSeqs[0]
    realLen = len(firstBatch)
    trimmedAttentionsFirstBatch = attentionFirstBatch[:, :realLen, :realLen]
    logger.logAttentions(trimmedAttentionsFirstBatch, firstBatch)

    tokeniser = Tokeniser(preprocesser, bpe)

    saveCheckpoint(gpt, config, merges, tokens, trainingPath)


    return gpt, tokeniser

