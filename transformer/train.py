from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from tokeniser.trainer import Trainer
from tokeniser.tokeniserutils import streamFileSymbols, tokeniseToDisk, countNonEmptyLines
from transformer.gpt import GPT
from transformer.transfomerutils import getLr
from transformer.memmapsampler import MemmapSampler
from tokeniser.tokeniser import Tokeniser
from transformer.config import Config
from transformer.checkpoint import saveCheckpoint, resumeCheckpoint
from traininglogger.logger import Logger
from itertools import islice
import tempfile
import os


import torch


def train(filePath: str, config: Config, trainingPath: str) -> tuple[GPT, Tokeniser]:

    torch.manual_seed(config.seed)

    preprocesser = PreProcessor()
    trainer = Trainer(islice(streamFileSymbols(filePath, preprocesser), 0, config.bpeTrainLines), config)
    merges, tokens = trainer.train_BPE()
    tokens.update(preprocesser.byteEncoder.values())
    bpe = BPE(merges, tokens)
    config.vocabSize = len(tokens)

    logger = Logger(config)
    gpt = GPT(config)
    gpt = gpt.to(config.device)

    parameterGroup1 = list(filter(lambda p: p.dim() >= 2, gpt.parameters()))
    parameterGroup2 = list(filter(lambda p: p.dim() < 2, gpt.parameters()))
    optimiser = torch.optim.AdamW(
      [{"params": parameterGroup1,   "weight_decay": config.weightDecay},
       {"params": parameterGroup2, "weight_decay": 0.0}],
      lr=config.maxLr)

    # Tokenise train/val splits to flat uint16 files on disk, then sample random
    # windows from them - bounded memory regardless of corpus size, see #12
    splitIndex = int(countNonEmptyLines(filePath) * (1 - config.valFraction))
    dataDir = tempfile.mkdtemp()
    trainPath = os.path.join(dataDir, "train.bin")
    valPath = os.path.join(dataDir, "val.bin")

    trainingSymbolStream = islice(streamFileSymbols(filePath, preprocesser), 0, splitIndex)
    tokeniseToDisk(trainingSymbolStream, bpe, trainPath)
    validationSymbolStream = islice(streamFileSymbols(filePath, preprocesser), splitIndex, None)
    tokeniseToDisk(validationSymbolStream, bpe, valPath)

    trainSampler = MemmapSampler(trainPath, config.maxLen)
    valSampler = MemmapSampler(valPath, config.maxLen)

    # Fixed once, up front, so every eval measures against the same data
    valBatches = [valSampler.sampleBatch(config.batchSize) for _ in range(config.numValBatches)]

    inputTensor = None

    # Resume if a checkpoint from an interrupted run is sitting at trainingPath
    startStep = 0
    if os.path.exists(trainingPath):
        startStep = resumeCheckpoint(trainingPath, gpt, optimiser)
        print(f"resuming from step {startStep:,}")

    for step in range(startStep, config.totalSteps):
        batch = trainSampler.sampleBatch(config.batchSize)
        inputTensor = batch[:, :-1].to(config.device)
        targetTensor = batch[:, 1:].to(config.device)

        # autocast covers the forward only - backward reuses the dtypes it recorded.
        # bf16 has fp32's exponent range so gradients cannot underflow, no GradScaler needed.
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16, enabled=config.useAmp):
            logits = gpt(inputTensor)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocabSize), targetTensor.reshape(-1))

        optimiser.zero_grad()
        loss.backward()
        gradNorm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        lr = getLr(step, config, config.totalSteps)
        for group in optimiser.param_groups:
            group["lr"] = lr
        optimiser.step()

        if step % config.logEvery == 0:
            logger.appendLoss(0, step, loss.item(), lr=lr, gradNorm=gradNorm.item())

        if step > 0 and step % config.checkpointEvery == 0:
            saveCheckpoint(gpt, config, merges, tokens, trainingPath, optimiser=optimiser, step=step)

        if step % config.evalEvery == 0:
            gpt.eval()
            with torch.no_grad():
                valTotalLoss = 0
                for valBatch in valBatches:
                    inputTensorv = valBatch[:, :-1].to(config.device)
                    targetTensorv = valBatch[:, 1:].to(config.device)
                    # same precision as training, or val loss would not be comparable
                    with torch.autocast(device_type=config.device, dtype=torch.bfloat16, enabled=config.useAmp):
                        logitsv = gpt(inputTensorv)
                        valLoss = torch.nn.functional.cross_entropy(logitsv.reshape(-1, config.vocabSize), targetTensorv.reshape(-1))
                    valTotalLoss += valLoss.item()
                logger.appendLoss(0, step, loss.item(), valTotalLoss / len(valBatches))

            gpt.train()

    if config.isDebug:
        attentions = [block.attention.attentionWeights for block in gpt.decoder.blocks]
        attentionFirstBatch = attentions[0][0]
        firstBatch = inputTensor[0].tolist()
        realLen = len(firstBatch)
        trimmedAttentionsFirstBatch = attentionFirstBatch[:, :realLen, :realLen]
        logger.logAttentions(trimmedAttentionsFirstBatch, firstBatch)

    tokeniser = Tokeniser(preprocesser, bpe)

    saveCheckpoint(gpt, config, merges, tokens, trainingPath, optimiser=optimiser, step=config.totalSteps)


    return gpt, tokeniser
