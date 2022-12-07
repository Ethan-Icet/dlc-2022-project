import torch


class Optimizer:
    def __init__(self, *parameters) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        pass


class SGD(Optimizer):
    def __init__(self, parameters: list, lr: float) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for _, g in self.parameters:
            g.zero_()

    def step(self) -> None:
        for param, grad in self.parameters:
            param -= self.lr * grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"
