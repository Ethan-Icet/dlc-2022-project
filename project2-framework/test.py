"""
You must implement a test executable named test.py that imports your framework and
• Generates a training and a test set of 1000 points sampled uniformly in [0, 1]^2, each with a
label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside,
• builds a network with two input units, one output unit, three hidden layers of 25 units,
• trains it with MSE, logging the loss,
• computes and prints the final train and the test errors.
"""
import torch
import math

from activation import  ReLU, Tanh, LossMSE
from linear import  Linear
from sequential import Sequential


def generate_points(x, y, pmin=0, pmax=1, nb=100):
	"""
		Function that generates nb points uniformely sampled in [pmin, pmax]^2
		returning balanced data set with x points and y labels
			 0 if outside the disk centered at (x, y) of radius 1/√2π 
			 1 otherwise
	"""
	radius = 1/math.sqrt(2*math.pi)
	
	center = torch.tensor([[0.5, 0.5]])
	
	x = torch.empty(nb, 2).fill_(0)
	y = torch.empty(nb).fill_(0)
	
	flag = 0
	inside = nb // 2 # nb of points inside the circle
	outside = nb - inside # nb of points outside the circle
	while not (inside + outside == 0) :
		z = torch.FloatTensor(1,2).uniform_(pmin, pmax)
	
		norm = torch.cdist(z, center, p=2)
	
		if norm > radius:
			if outside > 0:
				x[flag] = z
				outside -= 1
				flag += 1
		
		else:
			if inside > 0:
				x[flag] = z
				y[flag] = 1
				inside -= 1
				flag += 1

	return x, y


# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
	
# https://www.youtube.com/watch?v=pauPCy_s0Ok
	
# https://colab.research.google.com/drive/10y6glU28-sa-OtkeL8BtAtRlOITGMnMw
	
# multi - layer neural network	

if __name__ == '__main__':
	
	x_train, y_train = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=750)
	x_test, y_test = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=250)
	
	print('\ntrain',x_train.size(), y_train.size())
	print('test',x_test.size(), y_test.size())
	
	
	# define modèle
	n = 25
	seq = Sequential(Linear(2, n), ReLU(), Linear(n,n), ReLU(),Linear(n,n), Tanh(), Linear(n, 1), Tanh())
	
	# define mini-batch size
	
	for i in range(750):
	
		#for batch in ...:
		z = seq.forward(x_train[i].view(1, 2))
		
		seq.backward(z, y_train[i], None)
				# modify backward to cumulate gradients for batch calculation			
		# compute loss

		# implement optimizer while computing weights cf. course 5-2-SGD
	
	# write prediction function 1 if y > 0.5 ( - to define according to activation func-)
	
	z = seq.forward(x_test)
	z = torch.where(z > 0.0, 1, 0).t() # 0.5
	#print(z)

	print('\naccuracy', torch.sum(torch.where(y_test == z, 1, 0)) / y_test.size(0))
	
	n1 = torch.where(y_test == 1)
	n0 = torch.where(y_test == 0)

	print(torch.sum(torch.where(z.view(-1)[n1] == y_test[n1],1, 0)), '/', n1[0].size(0))
	print(torch.sum(torch.where(z.view(-1)[n0] == y_test[n0],1, 0)), '/', n0[0].size(0))
	
	
	
	
	
	
	
