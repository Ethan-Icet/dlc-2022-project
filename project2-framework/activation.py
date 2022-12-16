import torch
from module import *


class ReLU(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return torch.relu(x)

    def backward(self, x, grad, lr=0.01):
    	if (x == 0).any():
    		raise Exception('Undefined derivative for ReLU function.')
    		
    	return torch.where(x < 0, 0, 1) * grad # à vérifier selon calcul
	#return torch.where(x > 0, 1, 0)

    def param(self) -> list:
        return []

    def __repr__(self):
        return "ReLU()"


class Tanh(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return torch.tanh(x)

    def backward(self, x, grad, lr=0.01):
    	return (1 - torch.tanh(x)**2) * grad

    def param(self) -> list:
        return []

    def __repr__(self):
        return "Tanh()"


class LossMSE(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
    	#return (target-input)**2
        return torch.mean((target-input)**2)

    def backward(self, input, target):
    	return -2*(target-input)
    	#return torch.mean(-2*(target-input))

    def param(self) -> list:
        return []

    def __repr__(self):
        return "LossMSE()"
