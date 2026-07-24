import torch 
from transformer.gpt import GPT
from tokeniser.tokeniser import Tokeniser
from transformer.config import Config



def sampleNextToken(nextTokenLogits: torch.Tensor, config: Config) -> int:
    """
    Samples the next token id from logits using temperature and top-k sampling

    Args:
    nextTokenLogits: Tensor of shape (vocabSize,) - raw logits for the next token
    config: Config providing temperature (scaling applied before sampling - lower is
            more deterministic) and k (number of highest scoring tokens to sample from)

    Returns:
    nextToken: Sampled token id, as a plain int
    """

    scaledLogits = nextTokenLogits / config.temperature
    topValues, topIndices = torch.topk(scaledLogits, config.k)
    probs = torch.softmax(topValues, dim=-1)
    sampledIndex = torch.multinomial(probs, num_samples=1)
    return topIndices[sampledIndex].item()

def generate(gpt: GPT, tokeniser: Tokeniser, prompt: str, maxNewTokens: int, config: Config) -> str:
    """
    Generates text by autoregressively sampling new tokens from the model given a prompt

    Args:
    gpt: Trained GPT model
    tokeniser: Tokeniser used to encode the prompt and decode the generated ids
    prompt: Raw text to continue from
    maxNewTokens: Number of tokens to generate
    config: Config providing maxLen (context window fed into the model at each step),
            temperature and k (used for sampling - see sampleNextToken)

    Returns:
    Generated text, including the original prompt
    """
    encoded = tokeniser.encode(prompt)
    sequence = [tokenId for word in encoded for tokenId in word]

    with torch.no_grad():
        for _ in range(maxNewTokens):
            windowed = sequence[-config.maxLen:]
            logits = gpt([windowed])
            nextTokenLogits = logits[0, -1, :]
            nextToken = sampleNextToken(nextTokenLogits, config)
            sequence.append(nextToken)

    eowId = tokeniser.bpe.vocabMapping["/<w>"]
    
    words = []
    current = []

    for tokenId in sequence:
        current.append(tokenId)
        if tokenId == eowId:
            words.append(current)
            current = []

    if current:
        current.append(eowId)
        words.append(current)

    
    return tokeniser.decode(words)
    





