from module import *


class ReLU(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def param(self) -> list:
        return []

    def __repr__(self):
        return "ReLU()"


class Tanh(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def param(self) -> list:
        return []

    def __repr__(self):
        return "Tanh()"


class LossMSE(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
        pass

    def backward(self, input, target):
        pass

    def param(self) -> list:
        return []

    def __repr__(self):
        return "LossMSE()"
