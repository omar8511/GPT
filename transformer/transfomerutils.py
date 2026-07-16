import torch

def buildTrainingPairs(sequences: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Builds shift by one pairs for training

    Args:
    sequences: text encoded by BPE

    Returns:
    inputSeqs - list of words encoded by BPE dropping the last entry
    targetSeqs - list of words encoded by BPE dropping the first entr
    
    """
    inputSeqs = [seq[:-1] for seq in sequences]
    targetSeqs = [seq[1:] for seq in sequences]
    return inputSeqs, targetSeqs