from preprocessor import PreProcessor
from typing import Iterator

def mergePair(symbols: tuple[str], pair: tuple[str, str]):
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


def streamFileSymbols(file_path: str, preprocessor: PreProcessor) -> Iterator[list[list[str]]]:
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            
            yield preprocessor.tokenise(line)
             


    