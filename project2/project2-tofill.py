import math
import torch

from derivatives import *

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
	
	
class Module:
	
	def __init__(self, neurons=25):
		self.neurons = neurons # nb of neurons in hidden layer
		
		self.w1 = None # torch.empty(neurons,2)
		self.b1 = None # torch.empty(neurons,1)
		self.w2 = None # torch.empty(neurons,neurons)
		self.b2 = None # torch.empty(neurons,1)
		self.w3 = None # torch.empty(neurons,neurons)
		self.b3 = None # torch.empty(neurons,1)
		self.w4 = None # torch.empty(1,neurons)
		self.b4 = None # torch.empty(1,1)
		
	def init_weights(self):
		#w1 = torch.zeros(size=(self.neurons,2))
		w1 = torch.normal(0, 1e-06, size=(self.neurons,2))
		#w1 = torch.randint(0, 2, (neurons,2))
		self.w1 = w1.type(torch.FloatTensor)
		
		#b1 = torch.zeros(size=(self.neurons,1))
		b1 = torch.normal(0, 1e-06, size=(self.neurons,1))
		#b1 = torch.randint(10, 20, (neurons,1))
		self.b1 = b1.type(torch.FloatTensor)
		
		#w2 = torch.zeros(size=(self.neurons,self.neurons))
		w2 = torch.normal(0, 1e-06, size=(self.neurons,self.neurons))
		#w2 = torch.randint(0, 2, (neurons,neurons))
		self.w2 = w2.type(torch.FloatTensor)
		
		#b2 = torch.zeros(size=(self.neurons,1))
		b2 = torch.normal(0, 1e-06, size=(self.neurons,1))
		#b2 = torch.randint(10, 20, (neurons,1))
		self.b2 = b2.type(torch.FloatTensor)
		
		#w3 = torch.zeros(size= (self.neurons,self.neurons))
		w3 = torch.normal(0, 1e-06, size= (self.neurons,self.neurons))
		#w3 = torch.randint(0, 2, (neurons,neurons))
		self.w3 = w3.type(torch.FloatTensor)
		
		#b3 = torch.zeros(size= (self.neurons,1))
		b3 = torch.normal(0, 1e-06, size= (self.neurons,1))
		#b3 = torch.randint(10, 20, (neurons,1))
		self.b3 = b3.type(torch.FloatTensor)
		
		#w4 = torch.zeros(size= (1,self.neurons))
		w4 = torch.normal(0, 1e-06, size= (1,self.neurons))
		#w4 = torch.randint(0, 2, (1,neurons))
		self.w4 = w4.type(torch.FloatTensor)
		
		#b4 = torch.zeros(size= (1,1))
		b4 = torch.normal(0, 1e-06, size= (1,1))
		#b4 = torch.randint(10, 20, (1,1))
		self.b4 = b4.type(torch.FloatTensor)
		
		
	def forward(self, x):		
		z1 =  torch.mm(self.w1, x.t()) + self.b1
		a1 =  sigma1(z1)# activation function
		
		z2 =  torch.mm(self.w2, a1) + self.b2
		a2 =  sigma2(z2)# activation function
		
		z3 =  torch.mm(self.w3, a2) + self.b3
		a3 =  sigma3(z3)# activation function
	
		z4 =  torch.mm(self.w4, a3) + self.b4
		a4 =  sigma4(z4) #activation function
		
		return z1, a1, z2, a2, z3, a3, z4, a4
		
		
	def backward(self, a1, a2, a3, a4, z1, z2, z3, z4, x, y):		
		"""
			part to fill - compute gradients
		"""		
		return dw1, db1, dw2, db2, dw3, db3, dw4, db4
		
	def gradient_descent(self, w):
		"""
		with torch.no_grad():
			dw = torch.zeros(w.size())
			for i in range(w.size(0)):
				dw[i] = torch.gradient(w[i])[0]
				
		print('\ngrad',w.size(), dw.size())
		print(dw)"""	
		return 0
		
	def fit(self, x, y, learning_rate=0.01, epochs=100, batch_size=100, verbose = True):
		self.init_weights()
		
		n_mini_batch = x.size(0) // batch_size
		
		for epoch in range(epochs):
			e = torch.zeros(y.size()[0])
			
			for batch in range(n_mini_batch):
				u = torch.arange(batch*batch_size, (batch+1)*batch_size)
				x_train = x[u]
				y_train = y[u]
				
				for i, xi in enumerate(x_train):
					z1, a1, z2, a2, z3, a3, z4, a4 = self.forward(xi.view(1,2))
					#dw1, db1, dw2, db2, dw3, db3, dw4, db4 = self.backward(a1, a2, a3, a4, z1, z2, z3, z4, xi, y_train[i])
					
					
					#CUMULER GRAD ????
					"""
					# compute loss ( for a sample )
					e[i] = loss(a4[0], y_train[i])
					
					# backward - gradient
					self.w1 -= learning_rate * dw1
					self.w2 -= learning_rate * dw2
					self.w3 -= learning_rate * dw3
					self.w4 -= learning_rate * dw4.t()
					
					self.b1 -= learning_rate * db1
					self.b2 -= learning_rate * db2
					self.b3 -= learning_rate * db3
					self.b4 -= learning_rate * db4
					""""
					
				if verbose:   
					print('epoch=%d, batch=%d, lrate=%.3f, mse=%.3f' % (epoch, batch, learning_rate, torch.mean(e)))
			# compute MSE for all samples
			#mse = torch.mean(e)
			
			
	def predict(self, x):
		z1, a1, z2, a2, z3, a3, z4, a4 = self.forward(x)
		return torch.where(a4>0.0,1, 0)


if __name__ == "__main__":
	
	x_train, y_train = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=750)
	x_test, y_test = generate_points(0.5, 0.5, pmin=0, pmax=1, nb=250)
	
	print('\ntrain',x_train.size(), y_train.size())
	print('test',x_test.size(), y_test.size())
	
	module = Module(neurons=25)
	module.init_weights()
	
	module.fit(x_train, y_train, learning_rate=0.1, epochs=60, batch_size=50, verbose = False)
	
	z = module.predict(x_test)
	print('\naccuracy', torch.sum(torch.where(y_test == z, 1, 0)) / y_test.size(0))
	
	n1 = torch.where(y_test == 1)
	n0 = torch.where(y_test == 0)

	print(torch.sum(torch.where(z.view(-1)[n1] == y_test[n1],1, 0)), '/', n1[0].size(0))
	print(torch.sum(torch.where(z.view(-1)[n0] == y_test[n0],1, 0)), '/', n0[0].size(0))
	

