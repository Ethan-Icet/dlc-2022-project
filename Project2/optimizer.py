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
        self.t += 1

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


class Adam(Optimizer):

    def __init__(self, parameters: list, lr: float,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 amsgrad: bool = False, *, maximize: bool = False) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.m = [torch.zeros_like(param) for param, _ in self.parameters]  # first moment
        self.v = [torch.zeros_like(param) for param, _ in self.parameters]  # second moment
        self.m_hat = [torch.zeros_like(param) for param, _ in self.parameters]  # bias-corrected first moment
        self.v_hat = [torch.zeros_like(param) for param, _ in self.parameters]  # bias-corrected second moment
        self.t = 0  # number of steps
        if self.amsgrad:
            self.v_hat_max = [torch.zeros_like(param) for param, _ in self.parameters]  # second moment

    def zero_grad(self) -> None:
        for _, g in self.parameters:
            g.zero_()

    def step(self) -> None:
        # Based on the pseudocode from https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        for i, (param, grad) in enumerate(self.parameters):
            if self.maximize:
                grad = -grad
            if self.weight_decay != 0:
                grad += self.weight_decay * param
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * grad.pow(2)
            self.m_hat[i] = self.m[i] / (1 - self.betas[0] ** (self.t + 1))
            self.v_hat[i] = self.v[i] / (1 - self.betas[1] ** (self.t + 1))
            if self.amsgrad:
                self.v_hat_max[i] = torch.max(self.v_hat_max[i], self.v_hat[i])
                param -= self.lr * self.m_hat[i] / (self.v_hat_max[i].sqrt() + self.eps)
            else:
                param -= self.lr * self.m_hat[i] / (self.v_hat[i].sqrt() + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        s = f"Adam(lr={self.lr}"
        if self.betas != (0.9, 0.999):
            s += f", betas={self.betas}"
        if self.eps != 1e-8:
            s += f", eps={self.eps}"
        if self.weight_decay != 0:
            s += f", weight_decay={self.weight_decay}"
        if self.amsgrad:
            s += ", amsgrad=True"
        s += ")"
        return s
