import torch
from module import *


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return x.clamp(min=0)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        ds_dx = (torch.sign(self.x) + 1) / 2
        return ds_dx * grad

    def param(self) -> list:
        return []

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        # return 2 / (1 + (-2 * x).exp()) - 1
        return x.tanh()

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # ds_dx = 4 * torch.exp(2 * self.x) / (1 + torch.exp(2 * self.x)).pow(2)
        ds_dx = 1 - self.x.tanh().pow(2)
        return ds_dx * grad

    def param(self) -> list:
        return []

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return x.sigmoid()

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        ds_dx = self.x.sigmoid() * (1 - self.x.sigmoid())
        return ds_dx * grad

    def param(self) -> list:
        return []

    def __repr__(self) -> str:
        return "Sigmoid()"


class MSELoss(Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.target = None

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:
        # save the input to use it later in the backward pass
        self.input = input
        self.target = target
        out = (input - target).pow(2)
        if reduction == 'mean':
            out = out.mean()
        elif reduction == 'sum':
            out = out.sum()
        return out

    def backward(self, reduction: str = 'mean') -> torch.Tensor:
        out = (self.input - self.target)
        if reduction == 'mean':
            out = out / self.input.numel()
        # out has the shapes: (batch_size, n)
        return 2 * out

    def param(self) -> list:
        return []

    def __repr__(self) -> str:
        return "MSELoss()"


def _test():
    # Compare torch implementation with our own framework
    print("Test 1")
    shape = (5, 5)
    batch_size = 10
    x = torch.randn(batch_size, *shape)
    target = torch.randn(batch_size, *shape)
    relu = ReLU()
    tanh = Tanh()
    mse = MSELoss()
    y = relu.forward(x)
    z = tanh.forward(y)
    l = mse.forward(z, target)
    dl = mse.backward()
    dz = tanh.backward(dl)
    dy = relu.backward(dz)

    x.requires_grad = True
    y2 = torch.relu(x)
    z2 = torch.tanh(y2)
    l2 = torch.nn.functional.mse_loss(z2, target)
    l2.backward()
    dy2 = x.grad

    print("y", torch.allclose(y, y2), y, y2, sep='\n')
    print("z", torch.allclose(z, z2), z, z2, sep='\n')
    print("l", torch.allclose(l, l2), l, l2, sep='\n')
    print("dy", torch.allclose(dy, dy2), dy[0], dy2[0], sep='\n')

    print("Test 2")
    x = torch.randn(batch_size, *shape)
    reduction = 'mean'
    y = mse.forward(x, 2 * torch.ones(batch_size, *shape), reduction=reduction)
    dy = mse.backward(reduction=reduction)

    x.requires_grad = True
    y2 = torch.nn.functional.mse_loss(x, 2 * torch.ones(batch_size, *shape), reduction=reduction)
    y2.backward()
    dy2 = x.grad

    print("y", torch.allclose(y, y2), y, y2, sep='\n')
    print("dy", torch.allclose(dy, dy2), dy, dy2, sep='\n')


if __name__ == '__main__':
    _test()
