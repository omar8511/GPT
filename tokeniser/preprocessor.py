import regex


class PreProcessor:

    def __init__(self) -> None:
        safeIndices = set()
        SAFE_RANGE_ONE = [33, 127]
        SAFE_RANGE_TWO = [188, 256]

        self.byteEncoder = {} # Byte Level -> Unicode
        self.byteDecoder = {} # Unicode -> Byte Level
        regex_pattern = (
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
            r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )

        self.regex = regex.compile(regex_pattern)
        self.EOD = "<|endoftext|>"
        # SEP is ordinary text, not a special token: the model was pretrained
        # without it, so there is no embedding row for a reserved marker.
        # Byte-level BPE encodes any string, so this tokenises with the existing vocab.
        self.SEP = "\n\nStory:\n"

        for i in range(SAFE_RANGE_ONE[0], SAFE_RANGE_ONE[1]):
            safeIndices.add(i)

        for i in range(SAFE_RANGE_TWO[0], SAFE_RANGE_TWO[1]):
            safeIndices.add(i)

        shiftCounter = SAFE_RANGE_TWO[1]
        for i in range(SAFE_RANGE_TWO[1]):
            if i in safeIndices:
                self.byteEncoder[i] = chr(i)

            else:
                self.byteEncoder[i] = chr(shiftCounter)
                shiftCounter += 1

        # Unicode much bigger than utf8 which is just until 256


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

    def _utf8ToSafeUnicode (self, byteRepresentation: list[int]) -> list[str]:
        """
        Gives a Safe Unicode representation of the given UTF-8 encoding

        Parameters:
        word: UTF-8 level encoding to be converted to Safe Unicode characters 

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

        [["H", "E", "L", "L", "O"], ["B", "Y", "E"]]
    
        """
        segments = regex.split(
            rf"({regex.escape(self.EOD)})",
            text,
            )
        res = []
        # Split on EOD but keep it, then emit it atomically; everything else is ordinary text
        for segment in segments:
            if segment == self.EOD:
                res.append([segment])
            else:
                for word in self.regex.findall(segment):
                    res.append(self._utf8ToSafeUnicode(word.encode("utf-8")))

        return res


# Raw Text -> UTF-8 -> Safe UnicodeSymbols -> BPE UTF-8 allows you to encode any language

        
