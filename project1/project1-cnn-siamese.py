import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import matplotlib.pyplot as plt

from datetime import datetime

from dlc_practical_prologue import generate_pair_sets

"""###################################################################

	Siamese-CNN - to predict if a digit is greater than another
		input - 2 images from MNIST dataset
		output - corresponding label (0,1)

#######################################################################"""

# Function to compute the number of parameters in a model
def nb_parameters(model, trainable_only=True):
	if trainable_only:
		return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	else:
		return sum(parameter.numel() for parameter in model.parameters())



class Module(nn.Module):
	def __init__(self):
		super().__init__()
		"""
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*64, 256)
		self.fc2 = nn.Linear(256, 10)
		"""
		
		self.conv1 = nn.Conv2d (1, 8, kernel_size=3) #(1, 32, kernel_size=3) 
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3) #(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*16, 120) #(5*5*64, 256)
		self.fc2 = nn.Linear(120, 10) #(256, 10)
		
	def forward(self, x1):
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		#print(x.size())
		
		x = x.view(-1, 5*5*16 )#x.view(-1, 5*5*64 )
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		
		return x
		
		

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


	print(train_input[0].size())
	
	model = Module()
	
	criterion = nn.CrossEntropyLoss()#nn.MSELoss()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.0005, momentum = 0.85)  
	
	epochs = 10
	mini_batch_size = 50
	n_mini_batch = train_input.size(0) // mini_batch_size
	print('\nmini batch',mini_batch_size, n_mini_batch)
	
	acc = []
	
	start_time = datetime.now()
	for epoch in range(epochs):
		running_loss = 0.0

		for batch in range(n_mini_batch):
			# use of mini-batch
			u = torch.arange(batch*mini_batch_size, (batch+1)*mini_batch_size)
			x = train_input[u][:,0].view(mini_batch_size,1,14,14)
			y = train_classe[u][:,0]
			
			# zero the parameter gradient
			optimizer.zero_grad()
			
			# forward + backward + optimize
			outputs = model(x)
	
			loss = criterion(outputs, y)
			
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			#if batch % 10 == 9:    # print every 2000 mini-batches
				#print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
				#running_loss = 0.0
			
			_, p = torch.max(outputs, 1)
			#print(torch.sum(torch.where(p == y,1, 0)) / y.size(0))
			acc.append(torch.sum(torch.where(p == y,1, 0)) / y.size(0))
			
	end_time = datetime.now()
	print('\nTime:',end_time - start_time)
	print('Finished Training.\n')
	
	
	print(test_input.size(), test_classe.size())
	
	x = test_input[:,0].view(test_input.size(0),1,14,14)
	x2 = test_input[:,1].view(test_input.size(0),1,14,14)

	# predict
	outputs = model(x)
	_, p1 = torch.max(outputs, 1)
	
	outputs = model(x2)
	_, p2 = torch.max(outputs, 1)
	
	
	p = (p1 <= p2)
	p = torch.where(p == True, 1, 0)
	

	print(torch.sum(torch.where(p == test_target,1, 0)) / test_target.size(0))

	n1 = torch.where(test_target == 1)
	n0 = torch.where(test_target == 0)
	
	print(torch.sum(torch.where(p[n1] == test_target[n1],1, 0)), '/', n1[0].size(0))
	print(torch.sum(torch.where(p[n0] == test_target[n0],1, 0)), '/', n0[0].size(0))	

	
	print(f"\nnumber of parameters for the baseline model: {nb_parameters(Module())}")
	
	
	
	
	
	
	
	
	
	
	
	
	

