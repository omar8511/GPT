from tokeniser.preprocessor import PreProcessor
from tokeniser.bpe import BPE

class Tokeniser:

      def __init__(self, preProcessor: PreProcessor, bpe: BPE) -> None:
          self.preProcessor = preProcessor
          self.bpe = bpe

      def encode(self, text: str) -> list[list[int]]:
          """
          Encode raw text into BPE token ids.

          Args:
              text: Raw input string.

          Returns:
              Token ids grouped by pre-tokenised text chunk.
          """
          symbols = self.preProcessor.tokenise(text)
          return self.bpe.encode(symbols)

      def decode(self, tokenIds: list[list[int]]) -> str:
          """
          Decode BPE token ids back into raw text.

          Args:
              tokenIds: Token ids grouped by pre-tokenised text chunk.

          Returns:
              Reconstructed raw text.
          """
          return self.bpe.decode(tokenIds, self.preProcessor)