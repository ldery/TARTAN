import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_

EPS = 1e-8

# Calculates the dot products of 2 gradient vectors
def dot_prod(g1, g2):
	total = 0.0
	for p1, p2 in zip(g1, g2):
		if p1 is None or p2 is None:
			continue
		sum_ = (p1 * p2).sum()
		total += sum_
	total = total.item() if isinstance(total, torch.Tensor) else total
	return total

# Calculates the norm of a list of vectors
def calc_norm(list_of_vec):
	norm = 0.0
	for g_ in list_of_vec:
		if g_ is not None:
			norm += (g_**2).sum()
	return np.sqrt(norm.item())

# Calculates the cosine similarity
def cosine(vec_list_a, vec_list_b):
	a_norm = calc_norm(vec_list_a)
	b_norm = calc_norm(vec_list_b)
	cos_ = dot_prod(vec_list_a, vec_list_b) / (a_norm*b_norm + EPS)
	return cos_

# Creates a tensor of a specified shape and initializes all values to single scalar
def create_tensor(shape, init=0.0, requires_grad=True, is_cuda=True):
	inits = torch.ones(*shape) * init
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights


# Initialization function
def weight_init(init_method):
	def initfn(layer):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if init_method == 'xavier_unif':
				xavier_uniform_(layer.weight.data)
			elif init_method == 'kaiming_unif':
				kaiming_uniform_(layer.weight.data)
			elif init_method == 'kaiming_normal':
				kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				zeros_(layer.bias.data)
		elif isinstance(layer, nn.BatchNorm2d):
			layer.weight.data.fill_(1)
			layer.bias.data.zero_()
	return initfn