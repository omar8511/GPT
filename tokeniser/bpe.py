from datastructures.node import Node
from heapq import heappush, heappop
from tokeniser.preprocessor import PreProcessor

class BPE:
      
    def __init__(self, merges: list[tuple[str, str]], tokens: set[str]) -> None:
        self.mergeRanks = {merge : rank for rank, merge in enumerate(merges)}
        self.vocabMapping = {}
        self.integerMapping = {}
        self.encodeCache = {}
        tokens = sorted(tokens)
        for i, tok in enumerate(tokens):
            self.vocabMapping[tok] = i
            self.integerMapping[i] = tok
        

    def encode(self, text : list[list[str]]) -> list[list[int]]:
        """
        Returns BPE representation of tokenised text

        Args:
        text - Preprocessed list of symbols

        Returns:
        text - List of symbols in their BPE representation
        """

        toksWithEOW = [word + ["/<w>"] for word in text]        

        return [[self.vocabMapping[token] for token in symbols] for symbols in self._tokeniseWordSymbols(toksWithEOW)]
    
    def decode(self, mappings: list[list[int]], preProcessor: PreProcessor) -> str:
        """
        Provides the raw text of a BPE encoded string

        Args:
        mappings: BPE Encoding of a string
        preProcessor: Instance of a preprocessor

        Returns:
        Raw text of the encoded string
        """
        bpeToks = [[self.integerMapping[i] for i in mapping] for mapping in mappings]
        concatStrings = []
        for bpeTok in bpeToks:
            bpeTok.remove("/<w>")
            concatStrings.append("".join(bpeTok))

        byteIdsByWord = [[preProcessor.byteDecoder[c] for c in concatString] for concatString in concatStrings]

        byteIds = []
        for wordBytes in byteIdsByWord:
            byteIds.extend(wordBytes)

        return bytes(byteIds).decode("utf-8", errors="replace")

    def _tokeniseWordSymbols(self, wordSymbols: list[list[str]]) -> list[list[int]]:
        """
        Applies BPE merges to a list of words as symbols

        Args:
        wordSymbols: List of words as symbols

        Returns:
        res: List of symbols in their BPE representation
        """
        res = []
        for symbols in wordSymbols:
            res.append(self.helper(symbols))
        
        return res
        
    def helper(self, word: list[str]) -> list[str]: 
        """
        Applies priority merges to a word in its symbol representation 

        Args:
        word: word to have merges applied to 

        Returns:
        res: word with all merges applied in priority order
        
        """
        key = tuple(word)
        if key in self.encodeCache:
            return self.encodeCache[key]
        curr = Node(word[0])
        head = curr
        prev = None 
        heap = []

        for i in range(1, len(word)):
            newNode = Node(word[i], prev=curr)
            curr.next = newNode
            curr = newNode
            prev = newNode.prev

        curr = head
        counter = 0
        while curr.next:
            pair = (curr.val, curr.next.val)
            if pair in self.mergeRanks:
                heappush(heap, (self.mergeRanks[pair], counter, curr, curr.next))
                counter += 1
            curr = curr.next

        while heap:
            rank, _, leftNode, rightNode = heappop(heap)
            if leftNode.next != rightNode:
                continue
            newNode = Node(leftNode.val + rightNode.val)
            newNode.prev = leftNode.prev
            newNode.next = rightNode.next

            if leftNode.prev:
                leftNode.prev.next = newNode
                left_pair = (newNode.prev.val, newNode.val)
                if left_pair in self.mergeRanks:
                    heappush(heap, (self.mergeRanks[left_pair], counter, newNode.prev, newNode))
                    counter += 1
            else:
                head = newNode
                

            if rightNode.next:
                rightNode.next.prev = newNode
                right_pair = (newNode.val, newNode.next.val)
                if right_pair in self.mergeRanks:
                    heappush(heap, (self.mergeRanks[right_pair], counter, newNode, newNode.next))
                    counter += 1

            leftNode.prev = None
            leftNode.next = None
            rightNode.prev = None
            rightNode.next = None 


        curr = head
        res = []
        while curr:
            res.append(curr.val)
            curr = curr.next

        self.encodeCache[key] = res
        return res
    




        

        
