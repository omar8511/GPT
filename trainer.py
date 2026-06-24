from collections import Counter
from utils import mergePair

class Trainer:


    def __init__(self, text: str):
        self.corpus = text.split(" ")

    def train_BPE(self, numMerges: int) -> list[tuple[str, str]]:
        vocab = Counter()
        counts = Counter(self.corpus)
        for word in self.corpus:
            symbols = list(word) + ["_"]
            vocab[tuple(symbols)] += counts[word]

        merges = []

        for i in range(numMerges):
            
            pairCounts = Counter()

            for word, freq in vocab.items():
                symbols = word 
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pairCounts[pair] += freq 

            if not pairCounts:
                break
            
            bestPair = max(pairCounts, key = pairCounts.get)

            newVocab = Counter()

            for word, freq in vocab:
                newWord = mergePair(word, bestPair)
                newVocab[newWord] += freq
            
            vocab = newVocab
            merges.append(bestPair)

        return merges, vocab
    
    
    
    