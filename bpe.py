from utils import mergePair
from node import Node
from heapq import heappush, heappop

class BPE:
      
    def __init__(self, merges: list[tuple[str, str]], tokens: set[str]) -> None:
        self.mergeRanks = {merge : rank for rank, merge in enumerate(merges)}
        self.vocabMapping = {}
        tokens = sorted(tokens)
        for i, tok in enumerate(tokens):
            self.vocabMapping[tok] = i
        return
    
    def encode(self, text : list[list[str]]) -> list[list[int]]:
        """
        Appends EOW marker to each symbol representation

        Args:
        text - Preprocessed list of symbols

        Returns:
        text - List of symbols in their BPE representation
        """
        for t in text:
            t.append("/<w>")

        return self._tokeniseWordSymbols(text)

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
            else:
                head = newNode
                if left_pair in self.mergeRanks:
                    heappush(heap, (self.mergeRanks[left_pair], counter, newNode.prev, newNode))
                    counter += 1

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

        return res




        

        