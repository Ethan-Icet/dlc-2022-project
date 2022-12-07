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


class Basic(Optimizer):
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
        return f"Basic(lr={self.lr})"


class SGD(Optimizer):

    def __init__(self, parameters: list, lr: float,
                 momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0,
                 *, nesterov: bool = False, maximize: bool = False) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.v = [torch.zeros_like(param) for param, _ in self.parameters]  # momentum buffer
        self.t = 0  # number of steps

    def zero_grad(self) -> None:
        for _, g in self.parameters:
            g.zero_()

    def step(self) -> None:
        # Based on the pseudocode from https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        for i, (param, grad) in enumerate(self.parameters):
            if self.weight_decay != 0:
                grad += self.weight_decay * param
            if self.momentum != 0:
                if self.t > 0:
                    self.v[i] = self.momentum * self.v[i] + (1 - self.dampening) * grad
                else:
                    self.v[i] = grad
                if self.nesterov:
                    grad = grad + self.momentum * self.v[i]
                else:
                    grad = self.v[i]
            if self.maximize:
                param += self.lr * grad
            else:
                param -= self.lr * grad

    def __repr__(self) -> str:
        s = f"SGD(lr={self.lr}"
        if self.momentum != 0:
            s += f", momentum={self.momentum}"
        if self.weight_decay != 0:
            s += f", weight_decay={self.weight_decay}"
        if self.dampening != 0:
            s += f", dampening={self.dampening}"
        if self.nesterov:
            s += ", nesterov=True"
        s += ")"
        return s
