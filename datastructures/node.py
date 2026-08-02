from __future__ import annotations

class Node:

    def __init__(self, val: str, next: Node = None, prev: Node = None) -> None:
        self.val = val
        self.next = next
        self.prev = prev
