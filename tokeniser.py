from collections import Counter

class Tokeniser:


    def __init__(self, text: str):
        self.corpus = text.split(" ")

    def train_BPE(self, numMerges: int):
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
                newWord = self._mergePair(word, bestPair)
                newVocab[newWord] += freq
            
            vocab = newVocab
            merges.append(bestPair)

        return merges, vocab
    
    def encodeBPE(self, word: str, merges: list[tuple[str, str]]):
        symbols = list(word) + ["_"]

        for pair in merges:

            symbols = self._mergePair(symbols, pair)

        return symbols
    

    def _mergePair(self, symbols: tuple[str], pair: tuple[str, str]):
        result = []
        i = 0

        while i < len(symbols):
            if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair:
                result.append(symbols[i] + symbols[i+1])
                i += 2
            else:
                result.append(symbols[i])
                i += 1

        return tuple(result)


    