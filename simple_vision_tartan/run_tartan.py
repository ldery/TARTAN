from tartan_plugin import *
from data import *
from model import *
from torch.optim import Adam
import numpy as np
import random


def set_random_seed(seed=0):
	# Esp important for ensuring deterministic behavior with CNNs
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	cuda_available = torch.cuda.is_available()
	if cuda_available:
		torch.cuda.manual_seed_all(seed)
	return cuda_available

# Get the model
def get_model(output_classes_dict):
	model = WideResnet(output_classes_dict)
	model.cuda()
	return model

# Instantiate the optimizer
def get_optimizer(model, lr=1e-3):
	optimizer = Adam(model.parameters(), lr=lr)
	return optimizer

def run_epoch(model, dataset, optim, bsz, target_head='target', tartan_=None, mdev_head=None, dev_data=None):
	data_itr = iterator_(dataset[0], bsz)
	dev_head_itr = None if tartan_ is None else iterator_(dev_data, 16) # Iterator for fitting the dev head
	aux_itr = None if tartan_ is None else iterator_(dataset[1], bsz) # Iterator for the auxiliary data
	total_correct, total_egs, total_loss = 0.0, 0.0, 0.0

	for batch_ in data_itr:
		xs, ys = batch_
		model_outs = model(xs, head_name=target_head)
		loss_, acc_stats = model.criterion(model_outs, ys), model.accuracy(model_outs, ys)
		if aux_itr:
			# Get the target gradients
			target_grads = torch.autograd.grad(loss_, model.parameters(), allow_unused=True)

			# Get the auxiliary gradients
			# This is the simple case where there is only 1 auxiliary loss.
			# If there are n, auxiliary losses then we would need n iterators.
			try:
				aux_xs, aux_ys = next(aux_itr)
			except:
				aux_itr = iterator_(dataset[1], bsz)
				aux_xs, aux_ys = next(aux_itr)

			aux_loss_ = model.criterion(model(aux_xs, head_name='aux'), aux_ys)
			aux_grads = torch.autograd.grad(aux_loss_, model.parameters(), allow_unused=True)

			all_gradients = [target_grads, aux_grads]
			# Fit the dev-head
			if mdev_head:
				try:
					dev_batch = next(dev_head_itr)
					meta_batch = next(dev_head_itr)
				except:
					dev_head_itr = None if tartan_ is None else iterator_(dev_data, 16)
					dev_batch = next(dev_head_itr)
					meta_batch = next(dev_head_itr)

				mdev_head.learn_meta_dev_head(dev_batch, model.loss_fn)
				meta_grads = mdev_head.get_meta_grads(meta_batch, model.loss_fn)

				tartan_.update_weight_gradients(all_gradients, meta_grads)

			# Get the combined gradients
			combo_grads = tartan_.weighted_grads(all_gradients)
			# Set the combined gradients
			with torch.no_grad():
				for idx_, param in enumerate(model.parameters()):
					if combo_grads[idx_] is not None:
						param.grad = combo_grads[idx_]

			tartan_.update_weights()
		else:
			# There is no auxiliary data. Just do a normal backward
			loss_.backward()

		total_correct += acc_stats[0][0]
		total_egs += acc_stats[0][1]
		total_loss += loss_ *  acc_stats[0][1]
		if optim:
			optim.step()
			optim.zero_grad()

	return (total_loss / total_egs, total_correct / total_egs)

def graph_task_weights(task_names, weights, name):
	import matplotlib.pyplot as plt

	all_weights = np.stack(weights)
	for i_ in range(all_weights.shape[-1]):
		plt.plot(all_weights[:, i_], label=task_names[i_])
	plt.legend()
	plt.savefig("{}.png".format(name))


def run_tartan(model, data, task_names, bsz=64, n_epochs=30, is_meta_tartan=True):
	# Instantiate the optimizer and the tartan trainer
	optim = get_optimizer(model, lr=3e-4)
	tartan_trainer = TartanTrainer(task_names, is_meta_tartan=is_meta_tartan)
	meta_dev_head = None
	if is_meta_tartan:
		# Setup the dev head if doing meta-tartan
		dev_head = model.get_new_head(NUM_TARGET_CLASSES, 'dev_head')
		meta_dev_head = MetaDevHead(model, dev_head)

	dev_accs, test_accs = [], []
	for epoch in range(n_epochs):
		train_stats = run_epoch(model, data.train, optim, bsz, tartan_=tartan_trainer, mdev_head=meta_dev_head, dev_data=data.dev[0])

		dev_stats = run_epoch(model, data.dev, None, bsz)
		test_stats = run_epoch(model, data.test, None, bsz)
		print('Epoch {}: Train Loss {:.3f}, Train Ac {:.3f}'.format(epoch, *train_stats))
		print('        : Dev   Loss {:.3f}, Dev   Ac {:.3f}'.format(*dev_stats))
		print('        : Test  Loss {:.3f}, Test  Ac {:.3f}'.format(*test_stats))

		dev_accs.append(dev_stats[-1])
		test_accs.append(test_stats[-1])

	if is_meta_tartan:
		graph_task_weights(task_names, tartan_trainer.old_weights, 'meta-tartan')
	return test_accs[np.argmax(dev_accs)]

if __name__ == "__main__":
	data = CIFAR100_MSMammals()
	output_classes_dict = {
							'target': NUM_TARGET_CLASSES,
							'aux': NUM_AUX_CLASSES
					}

	set_random_seed()
	model = get_model(output_classes_dict)
	mttartan_testscore = run_tartan(model, data, ['target', 'aux'], is_meta_tartan=False)
	
	set_random_seed()
	model = get_model(output_classes_dict)
	metatartan_testscore = run_tartan(model, data, ['target', 'aux'], is_meta_tartan=True)

	print('Test Score | Multitasking-TARTAN   : {:.3f}'.format(mttartan_testscore))
	print('Test Score | Meta-Learning-TARTAN  : {:.3f}'.format(metatartan_testscore))