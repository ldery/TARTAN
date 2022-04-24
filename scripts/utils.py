import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

EPS = 1e-8

# For collating text data into batches and padding appropriately
def collate(examples, pad_token_id):
	return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)

def collate_fn(pad_token_id):
	def this_collate(examples):
		return collate(examples, pad_token_id)
	return this_collate

# Calculates the dot product between 2 gradient vectors
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
	norm_product = (a_norm * b_norm + EPS)
	cos_ = dot_prod(vec_list_a, vec_list_b) / norm_product
	return cos_

# Get the starting index of the heads of the network.
# This assumes that the network has naming such that {body_marker} substring is present in the names
# of all parameters that are part of the network body.
def get_body_end(model, body_marker='_text_field_embedder'):
	pos = 0
	for k, _ in model.named_parameters():
		if body_marker in k:
			pos += 1
	return pos

# Creates a tensor of a specified shape and initializes all values to single scalar
def create_tensor(shape, init=0.0, requires_grad=True, is_cuda=True):
	inits = torch.ones(*shape) * init
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights
