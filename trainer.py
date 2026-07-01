from collections import Counter
from utils import mergePair


class Trainer:


    def __init__(self, symbols: list[list[str]]) -> None:
        self.symbolList = symbols

    def train_BPE(self, numMerges: int) -> tuple[list[tuple[str, str]], set[str]]:
        """
        Perform numMerges iterations of BPE 

        Params:
        numMerges: Number of iterations of BPE to perform 

        Returns:
        merges: List of the pairs that were merged together
        tokens: Set of all tokens in the vocabulary
        """
        vocab = Counter()
        tokens = set()
        counts = Counter(tuple(s) for s in self.symbolList)

        for symbolTuple, count in counts.items():
            symbols = list(symbolTuple) + ["/<w>"]
            vocab[tuple(symbols)] += count
            tokens.update(symbols)

        merges = []

        for _ in range(numMerges):
            
            pairCounts = Counter()

            for word, freq in vocab.items():
                symbols = word 
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pairCounts[pair] += freq 

            if not pairCounts:
                break
            
            bestPair = max(pairCounts, key = pairCounts.get)
            newToken = bestPair[0] + bestPair[1]
            tokens.add(newToken)


            newVocab = Counter()

            for word, freq in vocab.items():
                newWord = mergePair(word, bestPair)
                newVocab[newWord] += freq
            
            vocab = newVocab
            merges.append(bestPair)

        return merges, tokens
    
    
    
    