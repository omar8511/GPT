from utils import mergePair

class BPE:
      
    def __init__(self):
        return

    def encodeBPE(self, word: str, merges: list[tuple[str, str]]):
        symbols = list(word) + ["_"]

        for pair in merges:

            symbols = mergePair(symbols, pair)

        return symbols