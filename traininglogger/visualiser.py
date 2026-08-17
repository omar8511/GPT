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
    valSteps = []
    valLosses = []
    lrSteps = []
    lrs = []
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

                    # validation only runs every evalEvery steps, so it needs its own x axis
                    valLoss = record.get("valLoss")
                    if valLoss is not None:
                        valSteps.append(step)
                        valLosses.append(valLoss)

                    lr = record.get("lr")
                    if lr is not None:
                        lrSteps.append(step)
                        lrs.append(lr)

                    if record["epoch"] != lastEpoch:
                        epochBoundaries.append(record["step"])
                        lastEpoch = record["epoch"]

        fig, ax = plt.subplots()
        ax.plot(steps, losses, label="train")

        if valLosses:
            ax.plot(valSteps, valLosses, label="validation")

            # the point where validation stops improving is where overfitting starts
            bestIdx = min(range(len(valLosses)), key=lambda i: valLosses[i])
            ax.axvline(x=valSteps[bestIdx], color="red", linestyle=":", alpha=0.8)
            ax.annotate(
                f"best val {valLosses[bestIdx]:.3f} @ step {valSteps[bestIdx]}",
                xy=(valSteps[bestIdx], valLosses[bestIdx]),
                fontsize=8,
            )

        for b in epochBoundaries:
            ax.axvline(x=b, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_yscale("log")  # loss spans orders of magnitude; log scale shows the convergence shape
        ax.set_title(f"Loss (batchSize={header['batchSize']}, numEpochs={header['numEpochs']})")
        ax.legend(loc="upper right")

        # learning rate shares the x axis but lives on a different scale entirely
        if lrs:
            lrAx = ax.twinx()
            lrAx.plot(lrSteps, lrs, color="green", alpha=0.4, linewidth=1)
            lrAx.set_ylabel("learning rate", color="green")
            lrAx.tick_params(axis="y", labelcolor="green")

        fig.savefig("docs/loss_curve.png")
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


