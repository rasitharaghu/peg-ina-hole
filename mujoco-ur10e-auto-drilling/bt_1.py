class Status:
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Node:
    def tick(self):
        raise NotImplementedError


class Sequence(Node):
    def __init__(self, children, name="Sequence"):
        self.children = children
        self.current_index = 0
        self.name = name

    def tick(self):
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick()

            if status == Status.SUCCESS:
                self.current_index += 1
                continue
            elif status == Status.RUNNING:
                return Status.RUNNING
            else:
                self.current_index = 0
                return Status.FAILURE

        self.current_index = 0
        return Status.SUCCESS


class Selector(Node):
    def __init__(self, children, name="Selector"):
        self.children = children
        self.current_index = 0
        self.name = name

    def tick(self):
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick()

            if status == Status.FAILURE:
                self.current_index += 1
                continue
            elif status == Status.RUNNING:
                return Status.RUNNING
            else:
                self.current_index = 0
                return Status.SUCCESS

        self.current_index = 0
        return Status.FAILURE


class BehaviorTree:
    def __init__(self, root):
        self.root = root

    def tick(self):
        return self.root.tick()
