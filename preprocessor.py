
import regex


class PreProcessor:

    def __init__(self) -> None:
        safeIndices = set()
        self.byteEncoder = {} # Byte Level -> Unicode
        self.byteDecoder = {} # Unicode -> Byte Level
        regex_pattern = (
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
            r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )

        self.regex = regex.compile(regex_pattern)

        for i in range(33, 127):
            safeIndices.add(i)

        for i in range(188, 256):
            safeIndices.add(i)

        shiftCounter = 256
        for i in range(256):
            if i in safeIndices:
                self.byteEncoder[i] = chr(i)

            else:
                self.byteEncoder[i] = chr(shiftCounter)
                shiftCounter += 1

        for k, v in self.byteEncoder.items():
            self.byteDecoder[v] = k
                


    def _unicodeToByte(self, word: str) -> list[int]:
        """
        Gives a byte representation of the given string

        Parameters:
        word: string to be converted

        Return:
        Byte representation of the word
        """

        res = []

        for c in word:
            res.append(self.byteDecoder[c])

        return res
    
    def _byteToUnicode (self, byteRepresentation: list[int]) -> list[str]:
        """
        Gives a Unicode representation of the given byte level encoding

        Parameters:
        word: Byte level encoding to be converted to Unicode 

        Return:
        Unicode representation 
        """

        res = []

        for i in byteRepresentation:
            res.append(self.byteEncoder[i])

        return res
        

    def tokenise(self, text: str) -> list[list[str]]:
        """
        Convert raw text to list of words as symbols

        Args:

        text: Raw input text

        Returns:
        List of words each represented as a list of symbols eg:

        [["H", "E", "L", "L", "O"], ["B", Y", "E"]]
    
        """

        words = self.regex.findall(text)
        res = []
        for word in words:
            byteLevel = self._unicodeToByte(word)
            res.append(self._byteToUnicode(byteLevel))

        return res


        