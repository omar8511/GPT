from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE
from typing import Iterator


def mergePair(symbols: tuple[str], pair: tuple[str, str]):
        """
        Merge all instances of a pair in a word

        Args:
        symbols: Representation of the word as a tuple of its characters
        pair: The pair to be merged

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


def streamFileSymbols(filePath: str, preprocessor: PreProcessor, chunkLength: int = 256) -> Iterator[list[list[str]]]:
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
             

def buildTrainingSequences(wordTokenLists: list[list[int]], maxLen: int) -> list[list[int]]:
      """
      Flatten BPE word level encodings into maxLen chunks

      Args:
      wordTokenLists - BPE word level encodings
      maxLen - length of each chunk

      Returns
      List of tokens each as maxLen chunks not per word
      
      """
      flat = [tokenId for word in wordTokenLists for tokenId in word]
      return [flat[i:i+maxLen] for i in range(0, len(flat), maxLen)]

    
def streamTrainingSequences(filePath: str, preprocessor: PreProcessor, bpe: BPE, maxLen: int) -> Iterator[list[int]]:
    """
    Streams training-ready token sequences from a file, one chunk at a time

    Args:
    filePath: Path to the file containing raw text
    preprocessor: Instance of the preprocessor
    bpe: Instance of BPE with merges/tokens already learned
    maxLen: Maximum length of each yielded sequence

    Returns:
    Iterator over token id sequences, each of length at most maxLen and greater than 1
    """
    buffer = []
         
    for lineSymbols in streamFileSymbols(filePath, preprocessor):
        wordTokenLists = bpe.encode(lineSymbols)
        for wordToken in wordTokenLists:
            buffer.extend(wordToken)

        while len(buffer) >= maxLen + 1:
            res = buffer[:maxLen + 1]
            buffer = buffer[maxLen + 1:]
            yield res

def batchSequences(sequenceStream: Iterator[list[int]], batchSize: int) -> Iterator[list[list[int]]]:
    """
    Groups a stream of token sequences into batches

    Args:
    sequenceStream: Iterator over individual token id sequences
    batchSize: Number of sequences per batch

    Returns:
    Iterator over batches, each a list of up to batchSize sequences
    """
    batch = []
    for sequence in sequenceStream:
        batch.append(sequence)
        if len(batch) == batchSize:
            yield batch
            batch = []
    if batch:
        yield batch