import torchvision
import numpy as np
import torch

msize_mammals_classes = [34, 63, 64, 66, 75]
NUM_TARGET_CLASSES = len(msize_mammals_classes)


other_classes = set(range(100)) - set(msize_mammals_classes)
other_classes = list(other_classes)
NUM_AUX_CLASSES = len(other_classes)


def to_tensor(data):
	if not len(data[0]):
		return None, None
	data = [torch.stack(data[0]), torch.tensor(data[1])]
	if torch.cuda.is_available():
		data[0] = data[0].cuda()
		data[1] = data[1].cuda()
	return data

N_SMALL_TRAIN = 200
N_SMALL_DEV = 64

class CIFAR100_MSMammals(object):
	def __init__(self):
		save_path = "~/"
		normalize = torchvision.transforms.Normalize(
											mean=[0.4914, 0.4822, 0.4465],
											std=[0.2023, 0.1994, 0.2010]
										)
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								normalize
							])
		train = torchvision.datasets.CIFAR100(save_path, train=True, download=True, transform=tform)
		test = torchvision.datasets.CIFAR100(save_path, train=False, download=True, transform=tform)
		self.train = self.group_data(train, dev_split=0.2)
		self.test = self.group_data(test)


	def group_data(self, data_, dev_split=-1):
		target = [[], []]
		auxiliary = [[], []]
		if dev_split > 0:
			self.dev = ([[], []], [[], []])
		for x, y in data_:
			if y in msize_mammals_classes:
				split_proba = dev_split * (dev_split > 0)
				res = np.random.binomial(1, 1.0 - split_proba, 1)
				if res[0]:
					if len(target[0]) >= N_SMALL_TRAIN:
						continue
					target[0].append(x)
					target[1].append(msize_mammals_classes.index(y))
				else:
					if len(self.dev[0][0]) > N_SMALL_DEV:
						continue
					self.dev[0][0].append(x)
					self.dev[0][1].append(msize_mammals_classes.index(y))
			else:
				auxiliary[0].append(x)
				auxiliary[1].append(other_classes.index(y))
		return (target, auxiliary)

def iterator_(dataset, bsz, shuffle=True):
	n_egs = len(dataset[0])
	order = np.random.permutation(n_egs) if shuffle else list(range(n_egs))
	nchunkz = n_egs // bsz
	chunks = np.array_split(order, nchunkz)
	for this_chunk in chunks:
		data = [[], []]
		for id_ in this_chunk:
			data[0].append(dataset[0][id_])
			data[1].append(dataset[1][id_])
		tensors = to_tensor(data)
		yield tensors