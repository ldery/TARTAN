from abc import abstractmethod
from .utils import *
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import ast


def add_config_args(parser):
	parser.add_argument(
							'-weight-strgy', type=str, default='default',
							choices=['default', 'meta']
	)
	parser.add_argument('-init-val', type=float, default=1.0, help='Initial Task weightings')
	parser.add_argument('--meta-lr-weight', type=float, default=0.01, help='learning rate for meta-learning')


def get_alpha_generator(opts, prim_key, aux_keys):
	weight_strgy = opts.weight_strgy
	if weight_strgy == 'default':
		return DefaultWeighter(prim_key, aux_keys, init_val=opts.init_val)
	elif weight_strgy == 'meta':
		return MetaWeighter(prim_key, aux_keys, meta_lr_weight=opts.meta_lr_weight)
	else:
		assert 'Invalid Value for Weighting strategy - {}'.format(weight_strgy)


class Weighter(object):
	def __init__(self, prim_key, aux_keys, init_val=1.0):
		self.weights = {key: init_val for key in aux_keys}
		self.aux_keys = aux_keys
		self.weights[prim_key] = init_val
		self.prim_key = prim_key
		self.init_val = init_val
		self.result_logs = []
		self.class_norm_logs = []
		self.is_meta = False

	def __getitem__(self, key):
		return self.weights[key]

	@abstractmethod
	def prep_epoch_start(self, epoch, **kwargs):
		pass

	def record_epoch_end(self, epoch, val_stat,  test_stat, **kwargs):
		entry = [self[k].item() if isinstance(self[k], torch.Tensor) else self[k] for k in self.weights.keys()]
		if 'class_norms' in kwargs:
			class_norm_entry = [v for _, v in kwargs['class_norms'].items()]
		else:
			class_norm_entry = [1.0]*len(self.weights)
		self.class_norm_logs.append(class_norm_entry)
		# Place the statistic to record in the final position
		entry.extend([val_stat, test_stat])
		self.result_logs.append(entry)

	def viz_results(self, save_loc, group_aux=True):
		to_viz_classnorms = np.array(self.class_norm_logs)
		to_viz = np.array(self.result_logs)
		all_keys = list(self.weights.keys())
		prim_idx = all_keys.index(self.prim_key)
		prim_vals = to_viz[:, prim_idx]
		fig, ax = plt.subplots(1, 2, figsize=(16, 8))
		ax[0].plot(range(len(prim_vals)), prim_vals, label='Primary Task Weighting')
		for idx_, key in enumerate(all_keys):
			if idx_ == prim_idx:
				desc = '{} Norm'.format(key) if not group_aux else 'Norm Per-Auxiliary Task'
				ax[1].plot(range(len(prim_vals)), to_viz_classnorms[:, idx_], linestyle='-.', label=desc)
				continue
			desc = '{}'.format(key) if not group_aux else 'Weight Per-Auxiliary Task'
			ax[0].plot(range(len(prim_vals)), to_viz[:, idx_], linestyle='dashed', label=desc)
			desc = '{} Norm'.format(key) if not group_aux else 'Norm Per-Auxiliary Task'
			ax[1].plot(range(len(prim_vals)), to_viz_classnorms[:, idx_], linestyle='-.', label=desc)
			if group_aux:
				break
		for i in range(2):
			ax[i].set_xlabel('Epoch')
			ax[i].set_ylabel('Weighting')
			ax[i].legend(loc='lower left')
		ax2 = ax[0].twinx()
		ax2.plot(range(len(prim_vals)), to_viz[:, -1], color='tab:red', label='Test Metric')
		ax2.plot(range(len(prim_vals)), to_viz[:, -2], color='tab:cyan', label='Val Metric')
		min_, max_ = np.min(to_viz[:, -2:]) - 0.01, np.max(to_viz[:, -2:]) + 0.01
		ax2.set_ylim(min_, max_)
		ax2.set_ylabel('Test/Val Metric', color='tab:red')
		ax2.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('{}/weighting_vrs_stat.png'.format(save_loc))


class MetaWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, meta_lr_weight=1e-2, init_=None):
		super(MetaWeighter, self).__init__(prim_key, aux_keys)
		# Finished setting up here
		# perform approriate intialization here
		all_classes = [prim_key, *aux_keys]
		self.weights = self.create_weights(all_classes, init=init_)
		self.meta_lr_weight = meta_lr_weight
		print('Starting learning with initial weights at : ', self.weights)
		self.is_meta = True

	def create_weights(self, classes, norm=1.0, requires_grad=True, init=None):
		if init is None:
			inits = np.array([1.0]* len(classes))
		else:
			assert len(init) == len(classes)
			inits = np.array(init)
		if norm > 0:
			inits = norm * inits / (sum(inits))
		# Create the weights
		weights = {class_: torch.tensor([inits[id_]]).float().cuda() for id_, class_ in enumerate(classes)}
		if requires_grad:
			for _, v in weights.items():
				v.requires_grad = True
		return weights

	# This gets the soft-max normalized version of the weight for a particular key
	def __getitem__(self, key):
		if not hasattr(self, 'sm_weights'):
			self.sm_weights = self.get_softmax()
		return self.sm_weights[key]
	
	# Do softmax on the meta-weights
	def get_softmax(self):
		with torch.no_grad():
			keys = []
			values = []
			for k, v in self.weights.items():
				keys.append(k)
				values.append(v)
			joint_vec = torch.cat(values)
			softmax = F.softmax(joint_vec, dim=-1)
		return {k: v for k, v in zip(keys, softmax)}
	
	# Set the gradients of the weights
	def set_weight_gradients(
								self, dev_task_grads, all_tasks_names, gradient_dict,
								body_params_end, dp_stats,
								scaling_factor, weight_stats
	):
		meta_weights = self.weights
		with torch.no_grad():
			dev_norm = calc_norm(dev_task_grads)  # Get the norm of the meta-dev gradients

			for task_id in all_tasks_names:
				# Core computation here
				task_norm = calc_norm(gradient_dict[task_id][:body_params_end])
				cos_sim = dot_prod(dev_task_grads, gradient_dict[task_id][:body_params_end])
				cos_sim = cos_sim / (dev_norm * task_norm)

				# Save state for visualization
				dp_stats[task_id].append(cos_sim)

				# Gradient is negative cosine similarity. Calculate and use that to set task weight gradients
				cos_sim = (torch.zeros_like(meta_weights[task_id]) - cos_sim)  / scaling_factor
				if meta_weights[task_id].grad is None:
					meta_weights[task_id].grad = cos_sim
				else:
					meta_weights[task_id].grad.add_(cos_sim)

				# Save state for visualization
				weight_stats[task_id].append((meta_weights[task_id].item(), meta_weights[task_id].grad.item(), dev_norm, task_norm, dp_stats[task_id][-1]))


	# Perform update on meta-related weights.
	def update_meta_weights(self):
		meta_weights = self.weights
		with torch.no_grad():
			for key in meta_weights.keys():
				if meta_weights[key].grad is None:
					meta_weights[key].grad = torch.zeros_like(meta_weights[key])
				new_val = meta_weights[key] - (self.meta_lr_weight * meta_weights[key].grad)
				meta_weights[key].copy_(new_val)
				meta_weights[key].grad.zero_()

		self.sm_weights = self.get_softmax()


class DefaultWeighter(Weighter):
	def __init__(self, prim_key, aux_keys, init_val=1.0):
		init_val = 1.0 / (1 + len(aux_keys))
		super(DefaultWeighter, self).__init__(prim_key, aux_keys, init_val=init_val)

	def __getitem__(self, key):
		return self.init_val

	def prep_epoch_start(self, epoch, **kwargs):
		pass

