import torch
from module import *

from activation import  ReLU, Tanh, LossMSE
from linear import  Linear

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
  

    def forward(self, x):
    	self.z = [x]
    	for module in self.modules:
    		x = module.forward(x)
    		self.z.append(x)
    	return x

    def backward(self, x, y, grad):
    	y = LossMSE().backward(x, y)   # verifier size, compute MSE
    	
    	#print('\n\n',self.modules, self.z) 
    	  	
    	for i, module in enumerate(self.modules[::-1]):
    		#print('\n\n',i, module, self.z[::-1][1:][i], y)
    		y = module.backward(self.z[::-1][1:][i], y, 0.01)
       

    def param(self) -> list:
        pass

    def __repr__(self):
        s = "Sequential(\n"
        for module in self.modules:
            s += str(module)
            s += ",\n"
        s += ")"
        return s
