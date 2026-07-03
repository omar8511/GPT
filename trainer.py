from collections import Counter
from utils import mergePair
from typing import Iterable


class Trainer:


    def __init__(self, symbolStream: Iterable[list[list[str]]]) -> None:
        self.vocab = Counter()
        self.tokens = set()

        for wordList in symbolStream:
            for wordSymbols in wordList:
                symbols = tuple(wordSymbols) + ("/<w>")

                self.vocab[symbols] += 1
                self.tokens.update(symbols)



    def train_BPE(self, numMerges: int) -> tuple[list[tuple[str, str]], set[str]]:
        """
        Perform numMerges iterations of BPE 

        Params:
        numMerges: Number of iterations of BPE to perform 

        Returns:
        merges: List of the pairs that were merged together
        tokens: Set of all tokens in the vocabulary
        """
        vocab = Counter(self.vocab)
        tokens = set(self.tokens)

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
    
    
    
    