import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dlc_practical_prologue import generate_pair_sets

"""##############################################################

	Simple CNN - to predict labels for MNIST dataset
		input - 1 image from MNIST dataset
		output - corresponding label

#################################################################"""


class Module(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*64, 256)
		self.fc2 = nn.Linear(256, 10)
		
	def forward(self, x1):
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
	
	test_input = z[0]
	print('\ntest_input', test_input.size())
	
	test_target = z[1]
	print('test_target', test_target.size())
	
	test_classe = z[2]
	print('test_classe', test_classe.size())

	
	model = Module()
	
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.0005, momentum = 0.85)  
	
	epochs = 10
	mini_batch_size = 50   #train_input.size(0) // epochs
	n_mini_batch = train_input.size(0) // mini_batch_size
	print('\nmini batch',mini_batch_size, n_mini_batch)
	
	
	for epoch in range(epochs):
		running_loss = 0.0
		
		#for i, data in enumerate(train_loader, 0):
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
			
	print('Finished Training.\n')
	
	
	print(test_input.size(), test_classe.size())
	
	x = test_input[:,0].view(test_input.size(0),1,14,14)
	y = test_classe[:,0]
	
	print(x.size(), y.size())

	# predict
	outputs = model(x)
	
	_, predicted = torch.max(outputs, 1)
	
	#print(predicted)
	
	print(torch.sum(torch.where(predicted == y,1, 0)) / y.size(0))
	
	n1 = torch.where(y == 1)
	n0 = torch.where(y == 0)
	
	print(torch.sum(torch.where(predicted[n1] == y[n1],1, 0)), '/', n1[0].size(0))
	
	print(torch.sum(torch.where(predicted[n0] == y[n0],1, 0)), '/', n0[0].size(0))
			

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

