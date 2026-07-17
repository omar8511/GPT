import torch 
from transformer.gpt import GPT
from tokeniser.tokeniser import Tokeniser


def sampleNextToken(nextTokenLogits: torch.Tensor, temperature: float, k: int) -> str:
    """
    Samples the next token id from logits using temperature and top-k sampling

    Args:
    nextTokenlogits: Tensor of shape (vocabSize,) - raw logits for the next token
    temperature: Scaling factor applied to logits before sampling - lower is more deterministic
    k: Number of highest scoring tokens to sample from

    Returns:
    nextToken: Sampled token id, as a plain int
    """

    scaledLogits = nextTokenLogits / temperature
    topValues, topIndices = torch.topk(scaledLogits, k)
    probs = torch.softmax(topValues, dim=-1)
    sampledIndex = torch.multinomial(probs, num_samples=1)
    return topIndices[sampledIndex].item()

def generate(gpt: GPT, tokeniser: Tokeniser, prompt: str, maxNewTokens: int, maxLen: int = 256, temperature: float = 1.0, k: int = 10) -> str:
    """
    Generates text by autoregressively sampling new tokens from the model given a prompt

    Args:
    gpt: Trained GPT model
    tokeniser: Tokeniser used to encode the prompt and decode the generated ids
    prompt: Raw text to continue from
    maxNewTokens: Number of tokens to generate
    maxLen: Maximum context window fed into the model at each step
    temperature: Scaling factor applied to logits before sampling - lower is more deterministic
    k: Number of highest scoring tokens to sample from at each step

    Returns:
    Generated text, including the original prompt
    """
    encoded = tokeniser.encode(prompt)
    sequence = [tokenId for word in encoded for tokenId in word]

    with torch.no_grad():
        for _ in range(maxNewTokens):
            windowed = sequence[-maxLen:]
            logits = gpt([windowed])
            nextTokenLogits = logits[0, -1, :]
            nextToken = sampleNextToken(nextTokenLogits, temperature, k)
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
    





