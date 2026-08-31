import torch
import json
import math
import random

from transformer.checkpoint import loadCheckpoint, saveCheckpoint
from transformer.transfomerutils import getLr
from traininglogger.logger import Logger
from sft.sftutils import collateBatch


def loadExamples(examplePath: str) -> list[tuple[str, str]]:
    """
    Reads instruction/response pairs from a JSONL file

    Args:
    examplePath: Path to a .jsonl file, one {"instruction": ..., "response": ...} per line

    Returns:
    List of (instruction, response) pairs
    """
    examples = []
    with open(examplePath, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                examples.append((record["instruction"], record["response"]))

    return examples


def sft(checkpointPath: str, examplePath: str, outPath: str, config) -> tuple:
    """
    Supervised fine-tuning: continues training a pretrained model on instruction/response
    pairs, scoring only the response tokens.

    Args:
    checkpointPath: Pretrained checkpoint to start from
    examplePath: JSONL of instruction/response pairs
    outPath: Where to write the fine-tuned checkpoint
    config: Config carrying the SFT hyperparameters (not the pretraining ones - see below)

    Returns:
    gpt, tokeniser
    """
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    gpt, tokeniser, savedConfig = loadCheckpoint(checkpointPath)
    config.vocabSize = savedConfig.vocabSize
    gpt = gpt.to(config.device)

    saved = torch.load(checkpointPath, weights_only=False, map_location="cpu")
    merges, tokens = saved["merges"], saved["tokens"]

    decayParams = [p for p in gpt.parameters() if p.dim() >= 2]
    noDecayParams = [p for p in gpt.parameters() if p.dim() < 2]
    optimiser = torch.optim.AdamW(
        [{"params": decayParams, "weight_decay": config.weightDecay},
         {"params": noDecayParams, "weight_decay": 0.0}],
        lr=config.maxLr)

    examples = loadExamples(examplePath)
    random.shuffle(examples)
    splitIndex = int(len(examples) * (1 - config.valFraction))
    trainExamples, valExamples = examples[:splitIndex], examples[splitIndex:]

    # Fixed once, so every eval measures against the same data
    valBatches = [
        collateBatch(valExamples[i:i + config.batchSize], config, tokeniser)
        for i in range(0, len(valExamples), config.batchSize)
    ]

    stepsPerEpoch = math.ceil(len(trainExamples) / config.batchSize)
    totalSteps = stepsPerEpoch * config.numEpochs
    logger = Logger(config)

    step = 0
    for epoch in range(config.numEpochs):
        random.shuffle(trainExamples)

        for i in range(0, len(trainExamples), config.batchSize):
            inputs, targets = collateBatch(trainExamples[i:i + config.batchSize], config, tokeniser)
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            with torch.autocast(device_type=config.device, dtype=torch.bfloat16, enabled=config.useAmp):
                logits = gpt(inputs)
                # -100 is cross_entropy's default ignore_index, so prompt and
                # padding positions are skipped and excluded from the mean
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, config.vocabSize), targets.reshape(-1))

            optimiser.zero_grad()
            loss.backward()
            gradNorm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            lr = getLr(step, config, totalSteps)
            for group in optimiser.param_groups:
                group["lr"] = lr
            optimiser.step()

            if step % config.logEvery == 0:
                logger.appendLoss(epoch, step, loss.item(), lr=lr, gradNorm=gradNorm.item())

            if step % config.evalEvery == 0:
                gpt.eval()
                with torch.no_grad():
                    valTotalLoss = 0
                    for valInputs, valTargets in valBatches:
                        valInputs = valInputs.to(config.device)
                        valTargets = valTargets.to(config.device)
                        with torch.autocast(device_type=config.device, dtype=torch.bfloat16, enabled=config.useAmp):
                            valLogits = gpt(valInputs)
                            valLoss = torch.nn.functional.cross_entropy(
                                valLogits.reshape(-1, config.vocabSize), valTargets.reshape(-1))
                        valTotalLoss += valLoss.item()
                    logger.appendLoss(epoch, step, loss.item(), valTotalLoss / len(valBatches))
                gpt.train()

            step += 1

    saveCheckpoint(gpt, config, merges, tokens, outPath, optimiser=optimiser, step=step)

    return gpt, tokeniser
