import torch
from sft.lora.LoRALinear import LoRALinear

def mergeLoRA(gpt):
    """
    Folds every LoRA adapter back into its base weight and restores plain nn.Linear layers.

    Args:
    gpt: Model whose LoRALinear layers should be merged

    Returns:
    The same model, with plain Linear layers and every parameter trainable again
    """
    targets = [n for n, m in gpt.named_modules() if isinstance(m, LoRALinear)]

    for path in targets:
        parentPath, _, attr = path.rpartition(".")
        parent = gpt.get_submodule(parentPath)
        loraLayer = getattr(parent, attr)

        merged = loraLayer.base
        with torch.no_grad():
            merged.weight += loraLayer.scale * (loraLayer.B @ loraLayer.A)
        setattr(parent, attr, merged)

    for p in gpt.parameters():
        p.requires_grad_(True)

    return gpt