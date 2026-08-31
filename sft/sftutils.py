import torch

from tokeniser.tokeniser import Tokeniser
from transformer.config import Config

def formatExample(instruction: str, response: str, tokeniser: Tokeniser) -> tuple[list[int], int]:
    """
    Formats a prompt and the example respone and returns an encoded tensor with the length of the prompt

    Args:
        instruction: The prompt
        response: The model response for that prompt
        tokeniser: Instance of Tokeniser

    Returns:
        tokTensor: list of tokens
        promptLength: length of the prompt
    """
    promptToks, responseToks = tokeniser.encode(instruction + tokeniser.preProcessor.SEP), tokeniser.encode(response)
    flattenedPToks, flattenedRToks = [tokenId for word in promptToks for tokenId in word], [tokenId for word in responseToks for tokenId in word], 
    return flattenedPToks + flattenedRToks, len(flattenedPToks)

def maskPrompt(input: list[int], promptLen: int) -> list[int]:
    """
    Apply a mask over the input and return the target array

    Args:
        input: list of tokens formed from the prompt, delimiter and response 
        promptLen: length of the prompt

    Returns:
        tokens: list of tokens with prompt tokens masked over to -100
    """
    target = input[1:] # To predict next tokens

    for i in range(promptLen - 1):
        target[i] = -100

    # -100 is a sentintel value for ignore_index in cross entropy
    return target

def collateBatch(batch: list[tuple[str, str]], config: Config, tokeniser: Tokeniser) -> torch.Tensor:
    batchLen = len(batch)
    formatted = [formatExample(i, r, tokeniser) for i, r in batch]
    inputs = [seq[:-1] for seq, _ in formatted]
    targets = [maskPrompt(seq, pLen) for seq, pLen in formatted]
    maxLen = max(len(t) for t in targets)

    inputs  = [s + [config.vocabSize]  * (maxLen - len(s)) for s in inputs]
    targets = [t + [-100]   * (maxLen - len(t)) for t in targets]

    return torch.tensor(inputs), torch.tensor(targets)
    
