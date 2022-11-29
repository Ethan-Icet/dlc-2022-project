from module import *


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def param(self) -> list:
        pass

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"
