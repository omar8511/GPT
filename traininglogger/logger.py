from transformer.config import Config
import os, json
from datetime import datetime
from dataclasses import asdict


class Logger:

    def __init__(self, config: Config) -> None:
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logDir = "logs"
        filePath = os.path.join(logDir, f"run_{timestamp}.jsonl")

        os.makedirs(logDir, exist_ok=True)
        self.file = open(filePath, "a")

        header = {"timestamp": timestamp, **asdict(config)}
        headerStr = json.dumps(header)

        self.file.write(headerStr + "\n")
        self.file.flush()

    def appendLoss(self, epoch: int, stepNumber: int, loss: float):
        """
        Appends a single loss record to the log file as a JSON line.

        Args:
        epoch: Current epoch number
        stepNumber: Global training step number
        loss: Loss value for this step
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lineJson = {"epoch": epoch, "step": stepNumber,"loss": loss, "timestamp": timestamp}
        line = json.dumps(lineJson)

        self.file.write(line + "\n")
        self.file.flush()