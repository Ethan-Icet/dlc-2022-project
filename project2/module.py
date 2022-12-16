import torch


class Linear():
	
	def __init__(self, x, y):
		self.input = x
		self.output = y # number of neurons
	
		#self.w = torch.normal(0, 1e-06, size=(self.output,self.input))
		#self.b = torch.normal(0, 1e-06, size=(self.output, 1))
		
		self.w = torch.randint(1, 7, size=(self.output,self.input))
		self.b = torch.randint(1, 7, size=(self.output, 1))
		
		self.w = self.w.type(torch.FloatTensor)
		self.b = self.b.type(torch.FloatTensor)
	
	def forward(self, x):
		print('\nw', self.w)
		print('\nb', self.b)
		x = x.type(torch.FloatTensor)
		return (torch.mm(self.w, x.t()) + self.b).t()
	
	def backward(self,):
		pass
	
	def params():
		pass
		


class Sequential():
	
	def __init__(self, *args):
		
		self.layers = []
		for arg in args:
			self.layers.append(arg)	
	
	def forward(self,x):
		for layer in self.layers:
			x = layer.forward(x)
		pass
	
	def backward(self, ):
		pass
	
	def params():
		pass
		
		

class Tanh():
	
	def __init__(self, ):
		pass
	
	def forward(self, x):
		return torch.tanh(x)
	
	def backward(self, x):
		return 1 - tanh(x)**2
	
	def params():
		pass
		

class ReLU():
	
	def __init__(self,):
		pass
	
	def forward(self, x):
		return torch.relu(x)
	
	def backward(self,):
		if (x == 0).any():
			#exit()
			raise Exception('Undefined derivative for ReLU function.')
	
		return torch.where(x < 0, 0, 1)
		#return torch.where(x > 0, 1, 0)
	
	def params(self,):
		pass
		
class MSE():
	
	def __init__(self,):
		pass
	
	def forward(self,):
		#return (y-x)**2
		return torch.mean((y-x)**2)
	
	def backward(self,):
		#return -2*(y-x)
		return torch.mean(-2*(y-x))
	
	def params(self,):
		pass
