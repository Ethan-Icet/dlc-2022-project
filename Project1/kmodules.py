import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import numpy as np

from datetime import datetime

from dlc_practical_prologue import generate_pair_sets

#from tools import *


class SiameseNN(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 16, kernel_size=3) # 1, 16
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # 16, 8
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*32, 160)
		self.fc2 = nn.Linear(160, 10)
		
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
	
	def forward(self, x):
		x1 = x[:,0].view(x.size(0),1,14,14)
		x2 = x[:,1].view(x.size(0),1,14,14)
		
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		
		a1, z1 = torch.max(y1, 1)
		a2, z2 = torch.max(y2, 1)
		
		x = torch.cat((z1.view(-1, z1.size(0)), z2.view(-1, z2.size(0))), axis=0).t()
		x = x.type(torch.FloatTensor)
		
		x = nn.functional.relu(self.lin1(x))
		x = nn.functional.relu(self.lin2(x))
		x = F.log_softmax(x, dim=1)
		
		return x, (y1, y2)


class SiameseNNAll(nn.Module):
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
	
	def forward(self, x):
		x1 = x[:,0].view(x.size(0),1,14,14)
		x2 = x[:,1].view(x.size(0),1,14,14)
		
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		
		a1, z1 = torch.max(y1, 1)
		a2, z2 = torch.max(y2, 1)
		
		x = torch.cat((z1.view(-1, z1.size(0)), z2.view(-1, z2.size(0))), axis=0).t()
		x = x.type(torch.FloatTensor)
		
		x = nn.functional.relu(self.lin1(x))
		x = nn.functional.relu(self.lin2(x))
		x = F.log_softmax(x, dim=1)
		
		return x, (y1, y2)


class Siamese(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d (1, 8, kernel_size=3) #(1, 32, kernel_size=3) 
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3) #(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*16, 120) #(5*5*64, 256)
		self.fc2 = nn.Linear(120, 10) #(256, 10)
		
	def forward_once(self, x1):
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		x = x.view(-1, 5*5*16 ) #(-1, 5*5*64 )
		
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		return x	
	
	def forward(self,x1, x2):
		
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		return None, (y1, y2)


class SiameseAll(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.maxpool_2d = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(5*5*64, 256)
		self.fc2 = nn.Linear(256, 10)
		
	def forward_once(self, x1):
		x = F.relu(self.conv1(x1))
		#x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		#x = F.dropout(x, p=0.5)#, training=training)
		
		x = x.view(-1, 5*5*64 )
		
		x = F.relu(self.fc1(x))
		#x = F.dropout(x, training=training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		return x	
	
	def forward(self,x):
		x1 = x[:,0].view(x.size(0),1,14,14)
		x2 = x[:,1].view(x.size(0),1,14,14)
		
		y1 = self.forward_once(x1)
		y2 = self.forward_once(x2) # torch.Size([50, 10])
		return None, (y1, y2)

class ContrastiveLoss(nn.Module):
	def __init__(self,):
		super(ContrastiveLoss, self).__init__()
		
	def forward(self, x1, x2, y):
		y1, z1 = torch.max(x1, 1)
		y2, z2 = torch.max(x2, 1)
		
		z = z2-z1 # y2*y1, z2-z1, y2-y1
		z = torch.minimum(torch.sign(z), torch.zeros(z.size())) + 1
	
		loss = torch.mean((y - z)**2) # mse
		#loss = torch.mean(abs(y-z))
	
		#yi = torch.where(y == z, 1, 0)
		#zi = y1*y2
		#zi = nn.functional.sigmoid(y2)*nn.functional.sigmoid(y1)
		#loss = torch.mean(yi * torch.log(zi) + (1-yi)*torch.log(1-zi)) # cross entropy
		
		return torch.tensor(loss, requires_grad = True)



	
	
	
	
	
	
	
	
	
	
	
	

