import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from dlc_practical_prologue import generate_pair_sets

"""###################################################################

	Siamese-CNN - to predict if a digit is greater than another
		input - 2 images from MNIST dataset
		output - corresponding label (0,1)

#######################################################################"""



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
	
	criterion = nn.CrossEntropyLoss()#nn.MSELoss()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.0005, momentum = 0.85)  
	
	# BATCH torch.Size([128, 1, 14, 14])
	
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
			x = train_input[u]
			y = train_classe[u]
			
			#print('\n',epoch, batch, u, x.size(), y.size())
			
			x = x[:,0].view(mini_batch_size,1,14,14)
			y = y[:,0]
			
			#print(x.size(), y.size())
			
			
			# zero the parameter gradient
			optimizer.zero_grad()
			
			# forward + backward + optimize
			outputs = model(x)
			
			#print('outputs',outputs.size())
			
			loss = criterion(outputs, y)
			
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
	print(x2.size(), y2.size())

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

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

