import torch

from module import Linear, Sequential, Tanh, ReLU, 

if __name__ == '__main__':
	print('hello')
	
	x = torch.tensor([[1,2]])
	
	a = Linear(2, 5)
	
	seq = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
	
	seq.forward(x)
