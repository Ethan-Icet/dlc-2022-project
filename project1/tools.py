import torch


from dlc_practical_prologue import generate_pair_sets



# Function to compute the number of parameters in a model
def nb_parameters(model, trainable_only=True):
	if trainable_only:
		return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	else:
		return sum(parameter.numel() for parameter in model.parameters())
		
		

def split(z):

	x = torch.cat((z[0], z[3]), dim = 0)
	y = torch.cat((z[1], z[4]), dim = 0)
	z = torch.cat((z[2], z[5]), dim = 0)
	
	i1 = torch.where(y == 1) # index for label 1
	i0 = torch.where(y == 0) # index for label 0
	
	"""
	p1 = torch.randperm(i1[0].size(0))
	p0 = torch.randperm(i0[0].size(0))
	
	n1 = p1.size(0) // 2 
	n0 = p0.size(0) // 2
	
	test = torch.cat((i1[0][p1[:n1]], i0[0][p0[:n0]]), dim=0)
	train = torch.cat((i1[0][p1[n1:]], i0[0][p0[n0:]]), dim=0)	
	"""
	p = torch.randperm(y.size(0))
	test = p[:y.size(0)//2]
	train = p[y.size(0)//2:]
	
	return x[train], y[train], z[train], x[test], y[test], z[test]

