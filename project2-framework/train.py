import math
import torch

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
	
	
	

if __name__ == "__main__":
	
	x_train, y_train = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=750)
	x_test, y_test = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=250)
	
	print('\ntrain',x_train.size(), y_train.size())
	print('test',x_test.size(), y_test.size())
	
	
	# define modèle
	n = 25
	seq = Sequential(Linear(2, n), Tanh(), Linear(n,n), Linear(n, 1))
	
	# define mini-batch size
	
	for i in range(20):
	
		y = seq.forward(x_train[i].view(1, 2))
		
		# predict if y > 0.5 or y < 0.5
		
		seq.backward(y, y_train[i], None)
				# modify backward to cumulate gradients 
				
		# compute loss

		# implement optimizer while computing weights cf. course 5-2-SGD
	
	# write prediction function 1 if y > 0.5 ( - to define according to activation func-)
	
	
	"""
	print('\naccuracy', torch.sum(torch.where(y_test == z, 1, 0)) / y_test.size(0))
	
	n1 = torch.where(y_test == 1)
	n0 = torch.where(y_test == 0)

	print(torch.sum(torch.where(z.view(-1)[n1] == y_test[n1],1, 0)), '/', n1[0].size(0))
	print(torch.sum(torch.where(z.view(-1)[n0] == y_test[n0],1, 0)), '/', n0[0].size(0))
	
	"""
