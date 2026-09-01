import torch

from sft.lora.LoRALinear import LoRALinear
from transformer.config import Config


def injectLoRa(gpt, config: Config):
    """
    Freezes a model and replaces every nn.Linear with a LoRA wrapped version.

    Args:
    gpt: Model to adapt

    Returns:
    The same model, with its Linear layers wrapped and everything else frozen
    """

    # Freeze first
    for p in gpt.parameters():
        p.requires_grad_(False)

    targets = [n for n, m in gpt.named_modules() if isinstance(m, torch.nn.Linear)]

    for path in targets:
        parentPath, _, attr = path.rpartition(".")
        parent = gpt.get_submodule(parentPath)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), config))

    return gpt
