import torch 
from transformer.gpt import GPT
from transformer.config import Config
from tokeniser.tokeniser import Tokeniser
from tokeniser.bpe import BPE
from tokeniser.preprocessor import PreProcessor


class Checkpoint(torch.nn.Module):

    def __init__() -> None:
        super().__init__()

    
    def saveCheckpoint(gpt: GPT, config: Config, merges: list[tuple[str, str]], tokens: set[str], path: str) -> None:
        """
        Saves model weights, config, and tokeniser data needed to reconstruct a working model

        Args:
        gpt: Trained GPT model
        config: Config used to build gpt, including vocabSize
        merges: Learned BPE merges
        tokens: Final BPE vocabulary
        path: File path to save the checkpoint to
        """
        checkpoint = {
            "state_dict": gpt.state_dict(),
            "config": config,
            "merges": merges,
            "tokens": tokens,
        }
        torch.save(checkpoint, path)


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
        checkpoint = torch.load(path)
        config = checkpoint["config"]

        bpe = BPE(checkpoint["merges"], checkpoint["tokens"])
        tokeniser = Tokeniser(PreProcessor(), bpe)

        gpt = GPT(config)
        gpt.load_state_dict(checkpoint["state_dict"])

        return gpt, tokeniser, config