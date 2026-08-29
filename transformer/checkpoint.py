import torch
import os
from transformer.gpt import GPT
from transformer.config import Config
from tokeniser.tokeniser import Tokeniser
from tokeniser.bpe import BPE
from tokeniser.preprocessor import PreProcessor




    
def saveCheckpoint(gpt: GPT, config: Config, merges: list[tuple[str, str]], tokens: set[str], path: str,
                   optimiser: torch.optim.Optimizer = None, step: int = None) -> None:
    """
    Saves model weights, config, and tokeniser data needed to reconstruct a working model.
    Passing optimiser and step additionally makes the checkpoint resumable - without the
    optimiser's moment estimates a resumed run effectively restarts Adam from cold.

    Args:
    gpt: Trained GPT model
    config: Config used to build gpt, including vocabSize
    merges: Learned BPE merges
    tokens: Final BPE vocabulary
    path: File path to save the checkpoint to
    optimiser: Optimiser whose state should be saved, for resuming
    step: Global step reached, so the LR schedule resumes mid-curve rather than re-warming
    """
    checkpoint = {
        "state_dict": gpt.state_dict(),
        "config": config,
        "merges": merges,
        "tokens": tokens,
        "step": step,
        "optimiser_state": optimiser.state_dict() if optimiser is not None else None,
        "rng_state": torch.get_rng_state(),
    }
    # write then rename, so a crash mid-write cannot leave a truncated checkpoint
    tempPath = path + ".tmp"
    torch.save(checkpoint, tempPath)
    os.replace(tempPath, path)


def loadCheckpoint(path: str) -> tuple[GPT, Tokeniser, Config]:
    """
    Loads a saved checkpoint, reconstructing the model and tokeniser

    Args:
    path: File path to load the checkpoint from

    Returns:
    gpt: GPT model with loaded weights
    tokeniser: Tokeniser rebuilt from the saved merges/tokens
    config: Config the model was trained with
    """
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint["config"]

    bpe = BPE(checkpoint["merges"], checkpoint["tokens"])
    tokeniser = Tokeniser(PreProcessor(), bpe)

    gpt = GPT(config)
    gpt.load_state_dict(checkpoint["state_dict"])

    return gpt, tokeniser, config


def resumeCheckpoint(path: str, gpt: GPT, optimiser: torch.optim.Optimizer) -> int:
    """
    Restores model, optimiser and RNG state in place from a checkpoint so training
    can continue where it stopped.

    Args:
    path: File path to load the checkpoint from
    gpt: Model to load weights into
    optimiser: Optimiser to load moment estimates into

    Returns:
    step: The global step to resume from, or 0 if the checkpoint predates step tracking
    """
    checkpoint = torch.load(path, weights_only=False)

    gpt.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("optimiser_state") is not None:
        optimiser.load_state_dict(checkpoint["optimiser_state"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])

    return checkpoint.get("step") or 0