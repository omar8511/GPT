from utils import mergePair

class BPE:
      
    def __init__(self, merges: list[tuple[str, str]], tokens: set[str]):
        self.merges = merges
        self.vocabMapping = {}
        tokens = sorted(tokens)
        for i, tok in enumerate(tokens):
            self.vocabMapping[tok] = i
        return

    def _tokeniseWord(self, word: str):
        symbols = list(word) + ["</w>"]

        for pair in self.merges:

            symbols = mergePair(symbols, pair)

        return symbols
    
    def encode(self, text: str):
        processed = text.split(" ")
        res = []
        
        for word in processed:
            for c in self._tokeniseWord(word):
                res.append(c)

        return [self.vocabMapping.get(tok, -1) for tok in res]

        

