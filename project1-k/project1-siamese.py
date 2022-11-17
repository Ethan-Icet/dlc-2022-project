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

###########################################################################"""

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
	
		self.fc3 = nn.Linear(20, 120)
		self.fc4 =  nn.Linear(120, 120)
		self.fc5 =  nn.Linear(120, 2)
		
		self.conv1 = nn.Sequential(         
		    nn.Conv2d(
		        in_channels=1,              
		        out_channels=16,            
		        kernel_size=5,              
		        stride=1,                   
		        padding=2,                  
		    ),                              
		    nn.ReLU(),                      
		    nn.MaxPool2d(kernel_size=2),    
		)
		self.conv2 = nn.Sequential(         
		    nn.Conv2d(16, 32, 5, 1, 2),     
		    nn.ReLU(),                      
		    nn.MaxPool2d(2),                
		)        # fully connected layer, output 10 classes
		self.out = nn.Linear(288, 10)
	
	def forward_once(self, x1):
		"""
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		x = x.view(-1, 5*5*64 )
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		"""
	
		x = self.conv1(x1)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		x = nn.Linear(x.size(1), 10)(x)
		return x	
	
	def forward(self,x1, x2):
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		
		x = torch.cat((y1,y2), axis=1)
		x = nn.Linear(20, 120)(x)
		x = F.relu(x)
		
		x = nn.Linear(120, 120)(x)
		x = F.relu(x)
		x = nn.Linear(120, 2)(x)
		#x = F.relu(x)
		#x = F.log_softmax(x, dim=1)
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
	
	test_input = z[0]
	print('\ntest_input', test_input.size())
	
	test_target = z[1]
	print('test_target', test_target.size())
	
	test_classe = z[2]
	print('test_classe', test_classe.size())

	
	model = Module()
	
	criterion = nn.CrossEntropyLoss()
	#criterion = nn.MSELoss()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.0005, momentum = 0.85)  

	
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
			loss1 = criterion(z1, y1)
			loss2 = criterion(z2, y2)
			
			loss3 = loss1 + loss2
			
			loss = criterion(z, y)
			
			#loss = 0.35 * loss + 0.65 * loss3
			
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

	
	print(torch.sum(torch.where(predicted == y,1, 0)) / y.size(0))
	
	n1 = torch.where(y == 1)
	n0 = torch.where(y == 0)
	
	print(torch.sum(torch.where(predicted[n1] == y[n1],1, 0)), '/', n1[0].size(0))
	
	print(torch.sum(torch.where(predicted[n0] == y[n0],1, 0)), '/', n0[0].size(0))
			

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

