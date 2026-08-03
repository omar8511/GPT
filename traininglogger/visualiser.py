import os
import json
import torch
import matplotlib.pyplot as plt

def visualiseLoss(path: str) -> None:
    """
    Reads a training log and plots the loss curve, with vertical markers at epoch boundaries.

    Args:
    path: Path to the .jsonl log file produced by Logger
    """
    steps = []
    losses = []
    epochBoundaries = []
    lastEpoch = None


    if os.path.isfile(path):
        with open(path, "r") as f:
            header = json.loads(f.readline())
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    step, loss = record["step"], record["loss"]
                    steps.append(step)
                    losses.append(loss)

                    if record["epoch"] != lastEpoch:
                        epochBoundaries.append(record["step"])
                        lastEpoch = record["epoch"]

        plt.plot(steps, losses, label="loss")

        for b in epochBoundaries:
            plt.axvline(x=b, color="gray", linestyle="--", alpha=0.5)

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.yscale("log")  # loss spans orders of magnitude; log scale shows the convergence shape
        plt.title(f"Training loss (batchSize={header['batchSize']}, numEpochs={header['numEpochs']})")
        plt.legend()
        plt.savefig("docs/loss_curve.png")
        plt.show()
    else:
        raise FileNotFoundError(f"No such log file: {path}")



def visualiseAttention(path: str) -> None:
    """
    Reads a saved attention snapshot and plots one heatmap per head, saved as a single PNG.

    Args:
    path: Path to the .pt file produced by Logger.logAttentions, containing
          "attentions" (numHeads, realLen, realLen) and "input" (token ids)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such attention file: {path}")

    data = torch.load(path)
    attentions = data["attentions"]
    tokens = data["input"]
    numHeads = attentions.shape[0]

    fig, axes = plt.subplots(1, numHeads, figsize=(4 * numHeads, 4))
    if numHeads == 1:
        axes = [axes]

    for headIdx in range(numHeads):
        ax = axes[headIdx]
        ax.imshow(attentions[headIdx].numpy())
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_title(f"head {headIdx}")

    fig.tight_layout()
    fig.savefig("docs/attention_heatmap.png")
    plt.show()


