from typing import List

class PreProcessor:

    def __init__():
        pass

    def tokenise(self, text: str) -> List[List[str]]:
        """
        Convert raw text to list of words as symbols

        Args:

        text: Raw input text

        Returns:
        List of words each represented as a list of symbols eg:

        [["H", "E", "L", "L", "O"], ["B", Y", "E"]]
    
        """

        words = text.split(" ")