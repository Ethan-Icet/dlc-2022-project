import torch
import math
from module import *
from activation import *
from linear import *
from sequential import *

"""
You must implement a test executable named test.py that imports your framework and
• Generates a training and a test set of 1000 points sampled uniformly in [0, 1]^2, each with a
label 0 if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside,
• builds a network with two input units, one output unit, three hidden layers of 25 units,
• trains it with MSE, logging the loss,
• computes and prints the final train and the test errors.
"""


# function to generate points inside and outside the circle
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
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    print('\ntrain',x_train.size(), y_train.size())
    print('test',x_test.size(), y_test.size())

    # define model
    n = 25
    model = Sequential(Linear(2, n), Tanh(), Linear(n,n),  ReLU(), Linear(n, 1), Tanh())
    lossMSE = MSELoss()

    # define mini-batch size
    batch_size = 10
    nb_epochs = 100
    lr = 0.01

    for e in range(nb_epochs):
        acc_loss = 0 # accumulate the loss
        # iterate over the number of mini-batches
        for b in range(0, x_train.size(0), batch_size):
            # get the mini-batch inputs and targets
            mini_batch_input = x_train.narrow(0, b, batch_size)
            mini_batch_target = y_train.narrow(0, b, batch_size)
            # compute the output and the loss for the mini-batch
            pred = model.forward(mini_batch_input)
            loss = lossMSE.forward(pred, mini_batch_target)
            # compute the accumulated loss
            acc_loss = acc_loss + loss.item()
            # gradient descent
            grad_loss = lossMSE.backward()
            grad_loss = model.backward(grad_loss)
            # update the parameters
            for p, g in model.param():
                p -= lr * g # update the parameters
                # reset the gradients to zero
                g.zero_()
            
        print(f"epoch: {e} loss: {acc_loss}", end="\r", flush=True)

    # compute the accuracy on the test set
    pred = model.forward(x_test)
    # compute the accuracy
    pred_label = torch.where(pred > 0.5, 1, 0)
    acc = torch.sum(pred_label == y_test).item() / y_test.size(0)
    print(f"\ntest accuracy: {acc}")
