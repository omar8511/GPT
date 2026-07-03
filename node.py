class Node:

    def __init__(self, val: str, next: Node = None, prev: Node = None) -> None:
        self.val = val
        self.next = next
        self.prev = prev
        pass

    def setPrev(self, prev: Node) -> None:
        self.prev = prev

    def setNext(self, next: Node) -> None:
        self.next = next