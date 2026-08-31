import torch
import math

from transformer.config import Config

def buildTrainingPairs(sequences: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Builds shift by one pairs for training

    Args:
    sequences: text encoded by BPE

    Returns:
    inputSeqs - list of words encoded by BPE dropping the last entry
    targetSeqs - list of words encoded by BPE dropping the first entry
    
    """
    inputSeqs = [seq[:-1] for seq in sequences]
    targetSeqs = [seq[1:] for seq in sequences]
    return inputSeqs, targetSeqs

def getLr(step: int, config: Config, totalSteps: int) -> float:
    """
    Learning rate for a given step: a linear warmup ramp followed by cosine decay

    Args:
    step: Current global training step
    config: Config providing maxLr (peak rate), minLr (floor the decay ends at)
            and warmupSteps (length of the ramp)
    totalSteps: Total steps the run will take, so the cosine finishes exactly at the end

    Returns:
    Learning rate to use for this step
    """
    if step < config.warmupSteps:
        return config.maxLr * (step + 1) / config.warmupSteps

    # 0 at the end of warmup, 1 at the end of training
    progress = (step - config.warmupSteps) / max(1, totalSteps - config.warmupSteps)
    progress = min(progress, 1.0)

    # cos goes 1 -> -1 over the run, so this factor goes 1 -> 0
    return config.minLr + 0.5 * (config.maxLr - config.minLr) * (1 + math.cos(math.pi * progress))

