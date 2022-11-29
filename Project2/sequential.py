from module import *


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def param(self) -> list:
        pass

    def __repr__(self):
        s = "Sequential(\n"
        for module in self.modules:
            s += str(module)
            s += ",\n"
        s += ")"
        return s
