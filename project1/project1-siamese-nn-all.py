import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import numpy as np

from datetime import datetime

from dlc_practical_prologue import generate_pair_sets

from tools import *

"""#######################################################################

	Siamese network - to predict if a digit is greater than another
			(use of nn to predict if digit is less than an other)
		input - 2 images from MNIST dataset
		output - corresponding label (0,1)
		
	- obtain better accuracy, but takes more time 
					and has more parameters-

###########################################################################"""
class Module(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 16, kernel_size=3) # 1, 16
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # 16, 8
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*32, 256)
		self.fc2 = nn.Linear(256, 10)
		
		self.lin1 = nn.Linear(2, 20)
		self.lin2 = nn.Linear(20, 2)
		
	def forward_once(self, x1):
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		x = x.view(-1, 5*5*32)
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		
		return x	
	
	def forward(self,x1, x2):
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		
		a1, z1 = torch.max(y1, 1)
		a2, z2 = torch.max(y2, 1)
		
		x = torch.cat((z1.view(-1, z1.size(0)), z2.view(-1, z2.size(0))), axis=0).t()
		x = x.type(torch.FloatTensor)
		
		x = nn.functional.relu(self.lin1(x))
		x = nn.functional.relu(self.lin2(x))
		x = F.log_softmax(x, dim=1)
		
		return x, y1, y2


if __name__ == "__main__":
	
	z = generate_pair_sets(1000)
	
	print('data set',len(z))	
	
	train_input = z[0]
	print('\ntrain_input', train_input.size())
	
	train_target = z[1]
	print('train_target', train_target.size())
	
	train_classe = z[2]
	print('train_classe', train_classe.size())
	
	test_input = z[3]
	print('\ntest_input', test_input.size())
	
	test_target = z[4]
	print('test_target', test_target.size())
	
	test_classe = z[5]
	print('test_classe', test_classe.size())


	print(f"\nnumber of parameters for the baseline model: {nb_parameters(Module())}")
	model = Module()
	
	criterion = nn.CrossEntropyLoss()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.0005, momentum = 0.85)  

	
	epochs = 25
	mini_batch_size = 20
	n_mini_batch = train_input.size(0) // mini_batch_size
	print('\nmini batch',mini_batch_size, n_mini_batch)
	
	start_time = datetime.now()
	for epoch in range(epochs):
		running_loss = 0.0
		
		for batch in range(n_mini_batch):
			u = torch.arange(batch*mini_batch_size, (batch+1)*mini_batch_size)
		
			x1 = train_input[u][:,0].view(mini_batch_size,1,14,14)
			y1 = train_classe[u][:,0]
			
			x2 = train_input[u][:,1].view(mini_batch_size,1,14,14)
			y2 = train_classe[u][:,1]
			
			y = train_target[u]
			
			# zero the parameter gradient
			optimizer.zero_grad()
			
			# forward + backward + optimize		
			z, z1, z2 = model(x1, x2)
		
			loss1 = criterion(z1, y1)
			loss2 = criterion(z2, y2)
			loss = criterion(z, y)
			
			loss3 = loss1 + loss2
			
			loss = loss + loss3 #(loss + loss3)/3
			
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			#if batch % 10 == 9:    # print every 2000 mini-batches
				#print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
				#running_loss = 0.0
			
	end_time = datetime.now()
	print('\nTime:',end_time - start_time)	
		
	print('Finished Training.\n')
	
	
	print(test_input.size(), test_target.size())
	
	x = test_input[:,0].view(test_input.size(0),1,14,14)	
	x2 = test_input[:,1].view(test_input.size(0),1,14,14)
	
	# predict
	predicted, z1, z2 = model(x, x2)	
	_, predicted = torch.max(predicted, 1)

	y = test_target
	
	print(torch.sum(torch.where(predicted == test_target,1, 0)) / test_target.size(0))
	
	n1 = torch.where(test_target == 1)
	n0 = torch.where(test_target == 0)
	
	print(torch.sum(torch.where(predicted[n1] == test_target[n1],1, 0)), '/', n1[0].size(0))
	print(torch.sum(torch.where(predicted[n0] == test_target[n0],1, 0)), '/', n0[0].size(0))
			

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

