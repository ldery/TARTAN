from tartan_utils import *
import torch.nn.functional as F
from torch.optim import AdamW
import pdb

'''
	Tartan Trainer Object
	parameters :
			task_names     : ordered lists of the names of the tasks being weighted
			weight_lr      : if we are using meta-learning, learning rate to use on the weights
			is_meta_tartan : whether we are using meta-tartan or mt-tartan
'''
class TartanTrainer(object):
	def __init__(self, task_names, weight_lr=5e-2, is_meta_tartan=True):
		self.is_meta_tartan = is_meta_tartan
		# Create task weights here. 
		self.task_names = task_names
		self.weights = create_tensor((len(task_names), ), init=0.0, requires_grad=is_meta_tartan, is_cuda=True)
		self.weight_lr = weight_lr if is_meta_tartan else 0.0
		self.old_weights = []

	# get the weightings corresponding to the different tasks.
	def get_weights(self, softmax=True):
		if softmax:
			with torch.no_grad():
				sm_ = F.softmax(self.weights, dim=-1)
				weights = sm_.cpu().numpy()
		else:
			weights = self.weights.cpu().numpy()
		return self.task_names, weights

	# perform gradient descent on the task weight logits
	def update_weights(self):
		if self.is_meta_tartan:
			with torch.no_grad():
				new_weights = self.weights - (self.weight_lr * self.weights.grad)
				self.weights.copy_(new_weights)
				self.weights.grad.zero_()

	# combine gradients according to the task weightings.
	# assumes that the order of (task_gradients) correspond to the order of  self.weights
	def weighted_grads(self, task_gradients):
		nparams = len(task_gradients[0])
		combo_grad = []
		softmax_ = self.get_weights()[-1]
		with torch.no_grad():
			for i_ in range(nparams):
				new_grad = None
				for j_ in range(len(self.task_names)):
					this_grad = task_gradients[j_][i_]
					if this_grad is not None:
						if new_grad is None: # Initialize to all zeros here
							new_grad = torch.zeros_like(this_grad)
						new_grad.add_(this_grad * softmax_[j_].item())
				combo_grad.append(new_grad)
		return combo_grad

	# compute the gradients of the weights and add this to the weight gradients
	def update_weight_gradients(self, task_gradients, dev_head_gradients):
		self.old_weights.append(self.get_weights()[-1])
		if self.is_meta_tartan:
			if self.weights.grad is None:
				self.weights.grad = torch.zeros_like(self.weights)
			with torch.no_grad():
				for i_ in range(len(task_gradients)):
					cos_sim = cosine(task_gradients[i_], dev_head_gradients)
					self.weights.grad[i_].add_(-cos_sim)