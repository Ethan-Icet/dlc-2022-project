import torch
from module import *


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features=  out_features # number of neurons
        
        self.w = torch.normal(0, 1e-06, size=(self.out_features,self.in_features))
        self.b = torch.normal(0, 1e-06, size=(self.out_features, 1))
     
        self.w = self.w.type(torch.FloatTensor)
        self.b = self.b.type(torch.FloatTensor)
        

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        return (torch.mm(self.w, x.t()) + self.b).t()

    def backward(self, x, grad, lr=0.01):
   
    	dw = x.t() * grad
    	db = grad * 1
    	dx= torch.mm(grad, self.w) # dx = torch.mm(grad, self.w.t())
    
    	# update weights
    	self.w -= lr * dw.t()
    	self.b -= lr * db.t()
    	return dx
   

    def param(self) -> list:
        pass

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"
