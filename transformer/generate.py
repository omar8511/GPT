import torch 
from transformer.gpt import GPT
from tokeniser.tokeniser import Tokeniser
from transformer.config import Config
from transformer.kvcache import KVCache


def sampleNextToken(nextTokenLogits: torch.Tensor, temperature: float = 1.0, k: int = 10) -> int:
    """
    Samples the next token id from logits using temperature and top-k sampling

    Args:
    nextTokenLogits: Tensor of shape (vocabSize,) - raw logits for the next token
    temperature: Scaling applied to the logits before sampling - lower is more
                 deterministic. Must be > 0, or the division produces NaNs
    k: Number of highest scoring tokens to sample from. Must be in 1..vocabSize,
       as torch.topk cannot take more entries than the tensor holds

    Returns:
    nextToken: Sampled token id, as a plain int
    """

    scaledLogits = nextTokenLogits / temperature
    topValues, topIndices = torch.topk(scaledLogits, k)
    probs = torch.softmax(topValues, dim=-1)
    sampledIndex = torch.multinomial(probs, num_samples=1)
    return topIndices[sampledIndex].item()

def generate(gpt: GPT, tokeniser: Tokeniser, prompt: str, maxNewTokens: int, config: Config, temperature: float = 1.0, k: int = 10) -> str:
    """
    Generates text by autoregressively sampling new tokens from the model given a prompt

    The prompt is run through the model in a single prefill pass, then tokens are generated
    one at a time. Each step feeds only the token just sampled and reuses the cached keys
    and values of every preceding position rather than recomputing them, so per-token cost
    stays flat instead of growing with the sequence.

    Args:
    gpt: Trained GPT model
    tokeniser: Tokeniser used to encode the prompt and decode the generated ids
    prompt: Raw text to continue from. A prompt longer than config.maxLen is truncated to
            its last maxLen tokens - the tokens nearest the generation point are the ones
            the model conditions on, and for the SFT format they include the delimiter
    maxNewTokens: Upper bound on tokens to generate. Fewer are returned if the context
                  window fills first, since positions beyond maxLen have no learned
                  positional embedding
    config: Config providing maxLen (context window, and the size of the positional
            embedding table), N (layer count - one cache per layer) and device
    temperature: See sampleNextToken
    k: See sampleNextToken

    Returns:
    Generated text, including the original prompt
    """
    encoded = tokeniser.encode(prompt)
    sequence = [tokenId for word in encoded for tokenId in word]
    cache = [KVCache() for _ in range(config.N)]
    gpt.eval()

    if len(sequence) > config.maxLen:
        sequence = sequence[-config.maxLen:]


    nextInput = sequence

    with torch.no_grad():
        for _ in range(maxNewTokens):
            if cache[0].length() >= config.maxLen:
                break

            logits = gpt(torch.tensor([nextInput]).to(config.device), cache = cache)
            token = sampleNextToken(logits[0, -1, :], temperature, k)
            sequence.append(token)
            nextInput = [token]

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

    gpt.train()

    
    return tokeniser.decode(words)
    





