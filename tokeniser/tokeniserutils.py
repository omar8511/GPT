from tokeniser.preprocessor import PreProcessor
from typing import Iterator

def mergePair(symbols: tuple[str], pair: tuple[str, str]):
        """
        Merge all instances of a pair in a word

        Args:
        symbols: Representation of the word as a tuple of its characters
        pairs: The pair to be merged

        Returns:
        result: Representation of the word with the pairs merged
        """
        if "/<w>" in pair:
             return symbols
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


def streamFileSymbols(filePath: str, preprocessor: PreProcessor) -> Iterator[list[list[str]]]:
        """
        Returns an iterator to the tokenised text

        Args:
        filePath: Path to the file containing raw text
        preprocessor: Instance of the preprocesser

        Returns:
        Iterator to the tokenised text
        """
        with open(filePath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                
                yield preprocessor.tokenise(line)
             


    
