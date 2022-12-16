import torch
import math

# https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd

sigma1 = lambda x: linear(x)
dsigma1 = lambda x: dlinear(x)

sigma2 = lambda x: tanh(x)
dsigma2 = lambda x: dtanh(x)

sigma3 = lambda x: tanh(x)
dsigma3 = lambda x: dtanh(x)

sigma4 = lambda x: tanh(x)
dsigma4 = lambda x: dtanh(x)


def linear(x):
	return x
def dlinear(x):
	return torch.ones(x.size())

def tanh(x):
	return torch.tanh(x)
		
def dtanh(x):
	return 1 - tanh(x)**2
		
def relu(x): 
	# reLu(x) = max(0,x)
	return torch.relu(x)
		
def drelu(x):
	"""
	if x < 0:
		return 0
	if x > 0:
		return 1
	"""
	if (x == 0).any():
		print('Undefined derivative of ReLU function.')
		exit()
		#raise Exception('Nahh!')
	
	return torch.where(x < 0, 0, 1)
	#return torch.where(x > 0, 1, 0)



def loss(x, y):
	return (y-x)**2
	#return torch.mean((y-x)**2)
	
def dloss(x, y):
	#return -2*(y-x)
	return torch.mean(-2*(y-x))
	
	
def forward(x, w1, b1, w2, b2, w3, b3, w4, b4):		
	z1 =  torch.mm(w1, x.t()) + b1
	a1 =  sigma1(z1)# activation function
		
	z2 =  torch.mm(w2, a1) #+ b2
	a2 =  sigma2(z2)# activation function
	
	z3 =  torch.mm(w3, a2) + b3
	a3 =  sigma3(z3)# activation function
	
	z4 =  torch.mm(w4, a3) + b4
	a4 =  sigma4(z4) #activation function
		
	return z1, a1, z2, a2, z3, a3, z4, a4
	


def derivative_a3(w4, z4, a4, y): # ok
	return dloss(y, a4) * dsigma4(z4) * w4
	
	
def derivative_a2(w3, w4, z3, z4, a4, y):
	da3 = derivative_a3(w4, z4, a4, y)
	da2 = torch.sum((da3 * dsigma3(z3).t()).t() * w3, axis=0)
	return da2.view(da2.size(0), 1)

def derivative4(z4, a3, a4, y): # ok
	dw4 = dloss(y, a4) * dsigma4(z4) * a3
	db4 = dloss(y, a4) * dsigma4(z4) * 1
	return dw4, db4
	
	
def derivative3(w4, z3, z4, a2, a4, y): # ok
	da3 = derivative_a3(w4, z4, a4, y)
	dw3 = (da3 * dsigma3(z3).t()).t() * a2.t()	
	db3 = (da3 * dsigma3(z3).t()).t() * 1
	return dw3, db3


def derivative2(w3, w4, z2, z3, z4, a1, a4, y): # ok
	da3 = derivative_a3(w4, z4, a4, y)
	a = torch.sum(da3.t() * dsigma3(z3) * w3 * dsigma2(z2).t(), axis = 0)
	dw2 = a.view(a.size(0),1) * a1.t()
	db2 = a.view(a.size(0),1) #* 1 
	return dw2, db2
	
	
def derivative1(w2, w3, w4, z1, z2, z3, z4, a4, x, y): # ok
	da2 = derivative_a2(w3, w4, z3, z4, a4, y)
	a = torch.sum(da2 * dsigma2(z2) * w2 * dsigma1(z1).t(), axis=0)
	dw1 = a.view(a.size(0),1)  * x
	db1 =  a.view(a.size(0),1)  #* 1
	return dw1, db1
	

"""
if __name__ == "__main__":
	
	x = torch.tensor([[1, 2]])
	y = torch.tensor([0.])
	
	x = x.type(torch.FloatTensor)
	y = y.type(torch.FloatTensor)
	
	# neurons 3
	
	# define manually weights
	w1 = torch.tensor([[1, 2], [2, 3], [1, 1]])
	b1 = torch.tensor([[7], [6], [4]])
	w1 = w1.type(torch.FloatTensor)
	b1 = b1.type(torch.FloatTensor)
	
	w2 = torch.tensor([[1, 5, 1], [2, 2, 3], [4, 3, 1]])
	b2 = torch.tensor([[1], [9], [7]])
	w2 = w2.type(torch.FloatTensor)
	b2 = b2.type(torch.FloatTensor)
	
	w3 = torch.tensor([[2, 3, 4], [3, 2, 1], [9, 2, 1]])
	b3 = torch.tensor([[1], [2], [2]])
	w3 = w3.type(torch.FloatTensor)
	b3 = b3.type(torch.FloatTensor)
	
	w4 = torch.tensor([[2, 2, 1]])
	b4 = torch.tensor([[1]])
	w4 = w4.type(torch.FloatTensor)
	b4 = b4.type(torch.FloatTensor)
	
	# compute forward
	z1, a1, z2, a2, z3, a3, z4, a4 = forward(x, w1, b1, w2, b2, w3, b3, w4, b4)
	
	print('\n\nz1', z1.size())
	print('a1',a1.size())
	print('z2',z2.size())
	print('a2',a2.size())
	print('z3',z3.size())
	print('a3',a3.size())
	print('z4',z4.size())
	print('a4',a4.size())


	da3 = derivative_a3(w4, z4, a4, y)
	print('\nda3', da3.size(), a3.size(), da3)
	
	da2 = derivative_a2(w3, w4, z3, z4, a4, y)
	print('\nda2', da2.size(), a2.size(), da2)
	
	dw4, db4 = derivative4(z4, a3, a4, y)
	print('\n\ndw4', dw4.t().size(), w4.size(), dw4.t())
	print('db4', db4.size(), b4.size(), db4)
	
	dw3, db3 = derivative3(w4, z3, z4, a2, a4, y)
	print('\ndw3', dw3.t().size(), w3.size(),'\n', dw3)
	print('db3', db3.size(), b3.size(), db3.t())

	dw2, db2 = derivative2(w3, w4, z2, z3, z4, a1, a4, y)
	print('\ndw2', dw2.t().size(), w2.size(),'\n', dw2)
	#print('db2', db2.size(), b2.size(), db2.t())
	
	dw1, db1 = derivative1(w2, w3, w4, z1, z2, z3, z4, a4, x, y)
	#print('\ndw1', dw1.t().size(), w1.size(),'\n', dw1)
	#print('db1', db1.size(), b1.size(), db1.t())
	
"""
	
	
	
