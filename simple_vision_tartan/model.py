from wideresnet import WideResNet
from tartan_utils import weight_init
import torch.nn as nn
import torch
from torch.optim import AdamW

class WideResnet(nn.Module):
	def __init__(
					self, output_classes_dict, depth=16, widen_factor=2, loss_name='CE', dropRate=0.1
				):
		super(WideResnet, self).__init__()
		self.model = WideResNet(depth, output_classes_dict, widen_factor=widen_factor, dropRate=dropRate)
		self.model.apply(weight_init('kaiming_normal'))
		self.loss_fn = nn.CrossEntropyLoss()
	
	def get_new_head(self, num_classes, head_name):
		return self.model.get_new_head(num_classes, head_name)

	def forward(self, x, head_name=None):
		m_out = self.model(x, head_name=head_name)
		return m_out

	def criterion(self, outs, target):
		return self.loss_fn(outs, target)

	def accuracy(self, outs, target):
		preds = torch.argmax(outs, dim=-1)
		ncorrect = (preds.eq(target)).sum().item()
		return (ncorrect, len(target)), ncorrect / (len(target) * 1.0)


# Wrapper around the model to perform computation on the dev-head
class MetaDevHead(nn.Module):
	def __init__(
					self, model_body, dev_head, head_init_fn=None,
					optim_wd=0.1, optim_lr=3e-2, optim_steps=10
	):
		super(MetaDevHead, self).__init__()
		self.model_body = model_body
		self.dev_head = dev_head
		self.head_init_fn = head_init_fn if head_init_fn else weight_init('kaiming_unif')
		self.optim_wd = optim_wd
		self.optim_lr = optim_lr
		self.optim_gd_steps = optim_steps

	# re-initialize the meta-dev head
	def reset_dev_head(self):
		self.dev_head.apply(self.head_init_fn)

	# Learn the meta-dev-head via gradient des
	def learn_meta_dev_head(self, dev_batch, dev_loss_fn):
		self.reset_dev_head()
		dev_head_params = self.dev_head.parameters()
		optim =  AdamW(dev_head_params, weight_decay=self.optim_wd, lr=self.optim_lr)
		dev_in, dev_out = dev_batch
		for j_ in range(self.optim_gd_steps):
			loss = dev_loss_fn(self.dev_head(self.model_body(dev_in)), dev_out)
			grads = torch.autograd.grad(loss, dev_head_params, allow_unused=True)

			# Populate the gradients
			with torch.no_grad():
				for p, g in zip(dev_head_params, grads):
					if p.grad is None:
						p.grad = torch.zeros_like(p)
					p.grad.add_(g)
					del g

			# Perform descent on only the dev-head
			optim.step()
			optim.zero_grad()

	# Get the gradients from the dev-head
	def get_meta_grads(self, dev_batch, dev_loss_fn):
		dev_in, dev_out = dev_batch
		loss = dev_loss_fn(self.dev_head(self.model_body(dev_in)), dev_out)
		all_params = list(self.model_body.parameters())
		grads = torch.autograd.grad(loss, all_params, allow_unused=True)
		return grads


