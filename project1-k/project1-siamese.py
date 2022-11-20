import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import numpy as np

from dlc_practical_prologue import generate_pair_sets

"""#######################################################################

	Siamese - to predict if a digit is greater than another
		(not working for the moment, need to be completed
			very low accuracy, model need to be changed )
			
	https://arxiv.org/pdf/2103.00200.pdf

###########################################################################"""

class Module(nn.Module):
	def __init__(self):
		super().__init__()
	
		self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
		
		self.pool1 = nn.MaxPool2d( kernel_size=2)#, stride=2)
		self.pool2 = nn.MaxPool2d( kernel_size=3)
		
		self.lin1 = nn.Linear(16, 64)
		self.lin2 = nn.Linear(64, 32)
		self.lin3 = nn.Linear(32, 10)
		"""
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*64, 256)
		self.fc2 = nn.Linear(256, 10)
		"""
	def forward_once(self, x1):
		
		x = self.pool1(nn.functional.relu(self.conv1(x1)))
		x = self.pool2(nn.functional.relu(self.conv2(x)))
		
		#print(x.shape)
		
		x = x.view(-1, x.size(1))
		
		x = nn.functional.relu(self.lin1(x))
		x = nn.functional.relu(self.lin2(x))
		x = self.lin3(x)
		
		"""
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		#print(x.size())
		
		x = x.view(-1, 5*5*64 )
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		"""
		return x	
	
	def forward(self,x1, x2):
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		
		x = torch.cat((y1,y2), axis=1)
		x = nn.Linear(20,40)(x)
		#x = F.sigmoid(x)
		
		x = nn.Linear(40, 2)(x)
		#x = F.sigmoid(x)
		x = F.log_softmax(x, dim=1)
		return x, y1, y2



class ContrastiveLoss(nn.Module):
	def __init__(self,):
		super(ContrastiveLoss, self).__init__()
		
	def forward(self, x1, x2, y):
		y1, z1 = torch.max(x1, 1)
		y2, z2 = torch.max(x2, 1)
		
		z = z2-z1 # y2*y1, z2-z1, y2-y1
		z = torch.minimum(torch.sign(z), torch.zeros(z.size())) + 1
		
		print(torch.sum(torch.where(z == y,1, 0)) / y.size(0))
		
		loss = torch.mean((y - z)**2) # mse
		
		loss = torch.mean(abs(y-z))
		
	
		#yi = torch.where(y == z, 1, 0)
		#zi = y1*y2
		#zi = nn.functional.sigmoid(y2)*nn.functional.sigmoid(y1)
		#loss = torch.mean(yi * torch.log(zi) + (1-yi)*torch.log(1-zi)) # cross entropy
		
		 #loss = - torch.sum(y * torch.log(y2*y1))
		
		return torch.tensor(loss, requires_grad = True)




if __name__ == "__main__":
	
	z = generate_pair_sets(1000)
	
	print('data set',len(z))	
	
	train_input = z[0]
	print('\ntrain_input', train_input.size())
	
	train_target = z[1]
	print('train_target', train_target.size())
	
	train_classe = z[2]
	print('train_classe', train_classe.size())
	
	test_input = z[0]
	print('\ntest_input', test_input.size())
	
	test_target = z[1]
	print('test_target', test_target.size())
	
	test_classe = z[2]
	print('test_classe', test_classe.size())

	
	model = Module()
	
	criterion1 = nn.CrossEntropyLoss()
	#criterion = nn.MSELoss()
	criterion = ContrastiveLoss()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.0005, momentum = 0.85)  

	
	epochs = 10
	mini_batch_size = 20   #train_input.size(0) // epochs
	n_mini_batch = train_input.size(0) // mini_batch_size
	print('\nmini batch',mini_batch_size, n_mini_batch)
	
	
	for epoch in range(epochs):
		running_loss = 0.0
		
		#for i, data in enumerate(train_loader, 0):
		for batch in range(n_mini_batch):
			# use of mini-batch
			u = torch.arange(batch*mini_batch_size, (batch+1)*mini_batch_size)
			x0 = train_input[u]
			y0 = train_classe[u]
		
			x1 = train_input[u][:,0].view(mini_batch_size,1,14,14)
			y1 = train_classe[u][:,0]
			
			x2 = train_input[u][:,1].view(mini_batch_size,1,14,14)
			y2 = train_classe[u][:,1]
			
			y = train_target[u]
			
			# zero the parameter gradient
			optimizer.zero_grad()
			
			# forward + backward + optimize		
			z, z1, z2 = model(x1, x2)
			
			#loss = criterion(x1, x2, y)
			loss1 = criterion1(z1, y1)
			loss2 = criterion1(z2, y2)
			#loss = criterion1(z, y)
			
			loss3 = loss1 + loss2
			
			loss = criterion(z1, z2, y)
			
			#loss = loss + 0.6*loss3
			
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			if batch % 10 == 9:    # print every 2000 mini-batches
				print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0
			
			
	print('Finished Training.\n')
	print(test_input.size(), test_classe.size())
	
	x = test_input[:,0].view(test_input.size(0),1,14,14)
	y = test_classe[:,0]
	
	x2 = test_input[:,1].view(test_input.size(0),1,14,14)
	y2 = test_classe[:,1]
	
	print(x.size(), y.size())

	# predict
	predicted, z1, z2 = model(x, x2)	
	_, predicted = torch.max(predicted, 1)
	
	#print(predicted)
	
	
	y1, z1 = torch.max(z1, 1)
	y2, z2 = torch.max(z2, 1)
	z = z2 - z1
	predicted = torch.minimum(torch.sign(z), torch.zeros(z.size())) + 1

	
	print(torch.sum(torch.where(predicted == y,1, 0)) / y.size(0))
	
	n1 = torch.where(y == 1)
	n0 = torch.where(y == 0)
	
	print(torch.sum(torch.where(predicted[n1] == y[n1],1, 0)), '/', n1[0].size(0))
	
	print(torch.sum(torch.where(predicted[n0] == y[n0],1, 0)), '/', n0[0].size(0))
			

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

