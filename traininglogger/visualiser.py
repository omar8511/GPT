import os
import json 
import matplotlib.pyplot as plt

def visualise(path: str) -> None:
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
        plt.title(f"Training loss (batchSize={header['batchSize']}, numEpochs={header['numEpochs']})")
        plt.legend()
        plt.savefig("loss_curve.png")
        plt.show()
    else:
        raise FileNotFoundError(f"No such log file: {path}")

