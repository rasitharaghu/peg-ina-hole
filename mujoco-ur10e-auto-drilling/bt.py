from enum import Enum, auto

class Status(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()

class Node:
    def tick(self):
        raise NotImplementedError

class Sequence(Node):
    def __init__(self, children, name="Sequence"):
        self.children = children
        self.index = 0
        self.name = name

    def tick(self):
        while self.index < len(self.children):
            status = self.children[self.index].tick()
            if status == Status.SUCCESS:
                self.index += 1
                continue
            if status == Status.FAILURE:
                self.index = 0
                return Status.FAILURE
            return Status.RUNNING
        self.index = 0
        return Status.SUCCESS

class Selector(Node):
    def __init__(self, children, name="Selector"):
        self.children = children
        self.index = 0
        self.name = name

    def tick(self):
        while self.index < len(self.children):
            status = self.children[self.index].tick()
            if status == Status.FAILURE:
                self.index += 1
                continue
            if status == Status.SUCCESS:
                self.index = 0
                return Status.SUCCESS
            return Status.RUNNING
        self.index = 0
        return Status.FAILURE

class BehaviorTree:
    def __init__(self, root):
        self.root = root

    def tick(self):
        return self.root.tick()
