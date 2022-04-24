import torch
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.modules import FeedForward
from allennlp.data import Vocabulary
from transformers import (
	AutoModelWithLMHead,
	AdamW,
)

import sys
import os

PATH=os.path.join(os.getcwd(), "dont_stop_pretraining")
sys.path.insert(1, PATH)
from models import BasicClassifierWithF1
from data.dataset_readers.text_classification_json_reader_with_sampling import TextClassificationJsonReaderWithSampling
from modules.seq2vec_encoders.cls_pooler import CLSPooler

from tqdm import tqdm
import numpy as np
import math
from .utils import *
from .alpha_generator import *

def add_tartan_args(parser):
	parser.add_argument('--dev_batch_sz', type=int, default=128, help='Batch sz for dev-set for meta-learning')
	parser.add_argument("--primary_task_id", type=str, default='citation_intent', choices=["citation_intent", "chemprot", "sciie", "hyperpartisan"])
	parser.add_argument("--classf_dev_lr", type=float, default=1e-4, help="Learning rate of dev-head")
	parser.add_argument("--classf_dev_wd", type=float, default=0.1)
	parser.add_argument("--classf_max_seq_len", type=int, default=512)
	parser.add_argument("--classf_iter_batchsz", type=int, default=8, help='Batch Size per iteration. True batch_sz is this x number of grad accumulation steps')
	parser.add_argument("--classifier_dropout", type=float, default=0.1)


class TartanModel(AutoModelWithLMHead):
	def __init__(
					self,
					model_name, # 'name of base lm model - eg - robertabase'
					base_lm_model,
					base_task_dataset_files,  # Dictionary of task_split : task_file

					max_seq_len=512,
					dropout=0.0,
					embedding_dim=768,
					ff_multiplier=1,
					num_layers=1,
					max_norm=1.0,

					batch_sz=8,
					save_path=None,
					primary_task_id=None,
					grad_accum_factor=8,
					dev_batch_sz=128,
	):
		assert save_path is not None, 'Invalid Save Path Provided for Classifier Head'
		assert isinstance(base_task_dataset_files, dict), 'Invalid type of base_task_dataset_files. Expected dictionary'
		assert primary_task_id is not None, 'No primary task id is given'
		assert 'train' in base_task_dataset_files, 'Primary Task not included in the list of dataset files'
		
		self.base_lm_model = base_lm_model
		self.file_for_split = base_task_dataset_files
		self.primary_task_id = primary_task_id
		prim_dataset_map = {self.primary_task_id: base_task_dataset_files['train']}
		self.datasets = self.setup_datasets(prim_dataset_map, model_name, max_seq_len, lazy=False)


		self.model_name = model_name

		# Cached for later use
		self.embedding_dim = embedding_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.ff_multiplier = ff_multiplier
		self.setup_classifier(
								dropout, self.primary_task_id, self.datasets[self.primary_task_id],
								embedding_dim, ff_multiplier, num_layers=num_layers
							)
		self.batch_sz = batch_sz
		self.max_norm = 1.0
		self.max_seq_len = max_seq_len

		self.save_path = save_path
		self.grad_accum_factor = grad_accum_factor
		self.aux_grads = defaultdict(list)
		self.dev_batch_sz = dev_batch_sz # Batch Size for dev-set
		self.label_vocab = None
		self.data_splits = defaultdict(lambda : None)

	'''
		The functionality below is for setting up for training.
	'''
	# Sets up the weighting generator
	def setup_alpha_generator(self, options):
		# Remove the primary task name if in list of auxiliary tasks
		aux_tasks = [x for x in options.aux_task_names if x != self.primary_task_id]
		self.aux_tasks = aux_tasks
		for x in self.aux_tasks:
			self.aux_grads[x] = None
		self.alpha_generator_algo = get_alpha_generator(options, self.primary_task_id, aux_tasks)
		# Setup datastructures for logging performance metrics
		if self.alpha_generator_algo.is_meta:
			self.options = options
			self.dp_stats = defaultdict(list)
			self.weight_stats = defaultdict(list)
			self.meta_head_perfs = defaultdict(list)


	# This method creates the classifier for a particular task.
	def setup_classifier(self, dropout, task_idx, dataset_dict, embedding_dim, ff_multiplier, num_layers=1):
		vocab = dataset_dict['vocab']
		text_field_embedder = self.base_lm_model
		seq2vec_encoder = CLSPooler(embedding_dim)
		hidden_dim = embedding_dim * ff_multiplier
		feedforward = FeedForward(
									embedding_dim, num_layers, hidden_dims=hidden_dim,
									activations=torch.nn.Tanh(), dropout=dropout
								)
		classifier = BasicClassifierWithF1(vocab, text_field_embedder, seq2vec_encoder, feedforward, dropout=dropout, initializer=None)
		classifier.to(self.base_lm_model.device)
		self.set_model(task_idx, classifier)
		return classifier

	# Move the model to the appropriate devices
	def to(self, device):
		for key in self.datasets.keys():
			this_classf = self.get_model(key)
			this_classf.to(device)
			# Since we have moved this to gpu, we need to re-set the base.
			this_classf._text_field_embedder = self.base_lm_model


	# This sets the optimizer and scheduler for further fine-tuning
	def set_optim(self, optimizer, scheduler):
		# Do this to set the optimizer
		self.optimizer = optimizer
		self.ft_lr_scheduler = scheduler

	'''
		House-keeping /utility functions defined below.
	'''

	# Save the model to the save path
	def save(self):
		path = self.save_path
		save_dict = {
				'optimizer_sd': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
				'scheduler': self.ft_lr_scheduler.state_dict() if hasattr(self, 'ft_lr_scheduler') else None,
				'perfs': dict(self.perfs) if hasattr(self, 'perfs') else None,
				'dp_stats': self.dp_stats if hasattr(self, 'dp_stats') else None,
				'weight_stats': self.weight_stats if hasattr(self, 'weight_stats') else None,
				'meta_head_perfs': self.meta_head_perfs if hasattr(self, 'meta_head_perfs') else None,
			}
		for key in self.datasets.keys():
			this_classf = self.get_model(key) 
			save_dict[key] = this_classf.state_dict()
		torch.save(
			save_dict,
			path
		)
	
	# Load the full model along with the optimizer and scheduler.
	def load(self):
		state_dict = torch.load(self.save_path)
		for key in self.datasets.keys():
			this_classf = self.get_model(key) 
			this_classf.load_state_dict(state_dict[key])

		if hasattr(self, 'optimizer') and ('optimizer_sd' in state_dict):
			self.optimizer.load_state_dict(state_dict['optimizer_sd'])
			self.ft_lr_scheduler = state_dict['scheduler']
		self.base_lm_model = this_classf._text_field_embedder


	# Load only the primary task model. Use this for the optional fine-tuning stage
	def load_primary(self, device):
		# We are assuming that what we care about is the primary task parameters
		primary_classifier = self.get_model(self.primary_task_id) 
		state_dict = torch.load(self.save_path) # Load the state dict from the save path
		primary_classifier.load_state_dict(state_dict[self.primary_task_id])
		primary_classifier.to(device)
		self.base_lm_model = primary_classifier._text_field_embedder
	
	# Sets the location for saving the model
	def set_save_path(self, save_path):
		self.save_path = save_path
	
	# Gets the classifier corresponding to a particular task name
	def get_model(self, model_name, check_=True):
		this_classf = getattr(self, "AuxHead-{}".format(model_name), None)
		if check_:
			assert this_classf is not None, 'Auxiliary Classifier {} not found'.format(model_name)
		return this_classf

	# Set the model corresponding to a particular task name
	def set_model(self, model_name, new_model):
		model_name = "AuxHead-{}".format(model_name)
		setattr(self, model_name, new_model)

	# Get metrics from a particular classifier. Defaults to the primary task classifier
	def get_metrics(self, this_classf=None, reset=False):
		if this_classf is None:
			this_classf = self.get_model(self.primary_task_id)
			# Get the metrics from the classifier
		return this_classf.get_metrics(reset=reset)

	# Get the list of classifier parameters
	def get_classifier_params(self, keys=None, withbase=False):
		param_list = []
		# Get all the classifier params if keys is not specified
		if keys is None:
			keys = self.datasets.keys()
		for _, key in enumerate(keys):
			this_classf = self.get_model(key)
			if withbase and key == self.primary_task_id:
				# This is the case where we need all the parameters for optional finetuning the primary task.
				param_list.extend(this_classf.named_parameters())
			else:
				# Removing the base RoBERTa model so we have only the added on classifier heads
				filtered_param_list = [param for pname, param in this_classf.named_parameters() if '_text_field_embedder' not in pname]
				param_list.extend(filtered_param_list)
		return param_list


	'''
		Functionality related to creating and learning the meta-dev head
	'''
	
	# This function leands a dev head for estimating the meta-gradients
	def learn_dev_head(self, sample_sz=-1):
		assert hasattr(self, 'options'), 'The options need to be set for training of the dev head'

		dev_head_name = "{}-{}".format('dev', self.primary_task_id)
		this_classf = self.get_model(dev_head_name, check_=False)

		if this_classf is None:
			# Need to instantiate the classifier head
			this_classf = self.setup_classifier(
								self.dropout, dev_head_name, self.get_dataset('dev'), # We use the dev-data to instantiate the classifier
								self.embedding_dim, self.ff_multiplier, num_layers=self.num_layers
							)
			# Setup optimizer for dev head
			dev_params = self.get_classifier_params([dev_head_name], withbase=False)
			dev_optim =  AdamW(
									dev_params, betas=eval(self.options.classf_betas),
									weight_decay=self.options.classf_dev_wd, lr=self.options.classf_dev_lr
								)
		else:
			return this_classf # We train this once and re-use
		
		# This is the first time instantiating this head so we need to train it
		assert dev_optim is not None, 'The optimizer for the dev head has not been instantiated'
		assert dev_params is not None, 'Dev Params should have been instantiated above'


		# perform gradient descent to get the dev-head
		sample_sz = sample_sz if sample_sz > 0 else self.dev_batch_sz
		samples = self.get_data_samples('dev', sample_sz)
		prev_loss_, tol = 1e10, 1e-3
		all_metrics = [[], [], []]

		for i in range(self.options.classf_ft_iters):
			output = this_classf(*samples)
			loss_ = output['loss']
			# This ensures that we only train the dev-head and keep the body fixed
			grads = torch.autograd.grad(loss_, dev_params, allow_unused=True)
			for p, g in zip(dev_params, grads):
				assert g is not None, 'This should have a gradient'
				p.grad = g
			dev_optim.step()

			# Save performance for analysis
			metrics = self.get_metrics(this_classf=this_classf, reset=True)
			all_metrics[0].append(metrics['f1'])
			all_metrics[1].append(metrics['accuracy'])
			all_metrics[2].append(loss_.item())

			if abs(loss_ - prev_loss_) < tol:
				break
			prev_loss_ = loss_.item()
			del grads
			torch.cuda.empty_cache()

		# Save performance for analysis
		self.meta_head_perfs['f1'].append(np.mean(all_metrics[0]))
		self.meta_head_perfs['accuracy'].append(np.mean(all_metrics[1]))
		self.meta_head_perfs['loss'].append(np.mean(all_metrics[2]))
		return this_classf

	# This function sets the dev_head
	def set_dev_and_get_grads_head(self):
		# Learn the meta-dev head here
		this_classf = self.learn_dev_head()

		# Get the dev gradient here
		dev_sent, dev_labels = self.get_data_samples('dev', self.batch_sz)
		try:
			loss_ = this_classf(dev_sent, dev_labels)['loss']
			gradients = torch.autograd.grad(loss_, this_classf.parameters(), allow_unused=True)
		except RuntimeError as e:
			if 'out of memory' in str(e):
				print('| WARNING: ran out of memory, retrying batch in set_dev_head')
				torch.cuda.empty_cache()
				loss_ = this_classf(dev_sent, dev_labels)['loss']
				gradients = torch.autograd.grad(loss_, this_classf.parameters(), allow_unused=True)
			else:
				raise e
		return gradients

	# This function resets the dev-head. We use this anytime we need a new approximation of the dev head
	def reset_dev_head(self):
		dev_head_name = "{}-{}".format('dev', self.primary_task_id)
		this_classf = self.get_model(dev_head_name)
		if this_classf is not None:
			del this_classf
		self.set_model(dev_head_name, None)


	'''
		Functionality for Training and Evaluating classifier
	'''

	# Get upstream gradients for the auxiliary tasks. These are MLM objectives in our case.
	def set_aux_grads(self, grads, aux_task_name='MLM'):
		if grads is not None:
			assert self.aux_grads[aux_task_name] is None, 'Need to make sure grads are none before setting'
		else:
			assert self.aux_grads[aux_task_name] is not None, 'Need to make sure grads are set before setting to none'
		self.aux_grads[aux_task_name] = grads

	
	# This method has two main functionalities
	# 1. Get gradients for the primary task
	# 2. If task weights are being meta-learned, compute the task weight gradients given (1.)
	def classifier_sample_grad(self):

		# Get the gradients w.r.t the primary task
		this_classf = self.get_model(self.primary_task_id)
		sent_dict, labels = self.get_data_samples('train', self.batch_sz)
		loss_ = this_classf(sent_dict, labels)['loss']
		prim_gradients = torch.autograd.grad(loss_, this_classf.parameters(), allow_unused=True)

		if self.alpha_generator_algo.is_meta:
			# We are doing meta-learning the task weights.
			gradient_dict = {}
			gradient_dict[self.primary_task_id] = prim_gradients

			# We only use the gradients of the shared body of the network for when updating the task weightings.
			if not hasattr(self, 'body_params_end'):
				self.body_params_end = get_body_end(this_classf)

			# Set the gradients of the other tasks involved
			for key, grad_list in self.aux_grads.items():
				# Get the current auxiliary task gradient.
				assert grad_list is not None, 'MLM Grads for {} should have been set by now'.format(key)
				gradient_dict[key] = grad_list[:self.body_params_end]

			# Get the dev-head gradient
			gradient_dict["dev-{}".format(self.primary_task_id)] = self.set_dev_and_get_grads_head()
			dev_task_grads = gradient_dict["dev-{}".format(self.primary_task_id)][:self.body_params_end]

			# Caculate the gradients of the task weights based on the dev-head
			# Todo [ldery] - move this code to a better location
			all_tasks_names = self.aux_tasks + [self.primary_task_id]
			self.alpha_generator_algo.set_weight_gradients(
																dev_task_grads, all_tasks_names, gradient_dict,
																self.body_params_end, self.dp_stats,
																self.grad_accum_factor, self.weight_stats
			)


		# Add the primary task gradients after scaling appropriately.
		with torch.no_grad():
			scaling = self.alpha_generator_algo[self.primary_task_id] / self.grad_accum_factor
			for idx, (p, g) in enumerate(zip(this_classf.parameters(), prim_gradients)):
				if p.grad is None:
					p.grad = torch.zeros_like(p)
				if g is None:
					continue
				p.grad.add_(scaling * g)
				del g
		del prim_gradients

	# Hack so things work easily with Don't Stop Pre-training code
	# Setup this models forward pass to respond only to the lm forward pass
	def forward(*args, **kwargs):
		# If we want forward pass to respond to specific head, then we have to run forward on model obtained from self.get_model(...)
		return self.base_lm_model(*args, **kwargs)

	# Evaluate the classifier
	def evaluate_classifier(self, set_='dev'):
		# Get the data and the classifier
		assert set_ in ['dev', 'test'], 'Wrong split specified'
		dataset = self.get_dataset(set_)
		this_classf = self.get_model(self.primary_task_id)

		torch.cuda.empty_cache()
		# reset the metrics before running new stuff
		try:
			_ = self.get_metrics(this_classf=this_classf, reset=True)
		except:
			print('This classifier does not need to reset metrics.')
		# Run the classifier
		this_classf.eval()
		with torch.no_grad():
			for samples in self.dataset_iterator(dataset, batchsz=self.batch_sz):
				_ = this_classf(*samples)
		this_classf.train()
		# Get the metrics from the classifier
		torch.cuda.empty_cache()
		return self.get_metrics(this_classf=this_classf, reset=True)

	# This code trains the primary head. This is for further finetuning after doing end-task aware training with auxiliary tasks
	def train_primary(self, n_iters, optimizer, lr_scheduler, max_grad_norm, patience=3, metric='f1'):

		best_iter, iters_since_improvement = 0, 0
		self.perfs = defaultdict(list)

		# Get the primary task model and training data
		prim_classf = self.get_model(self.primary_task_id)
		prim_classf.train()
		dataset = self.get_dataset('train')

		# Finetune for specified # of iterations
		for iter_ in range(n_iters):
			print('Currently on Classifier Epoch {}/{}'.format(iter_ + 1, n_iters))
			iterator = self.dataset_iterator(dataset, shuffle=True)
			total_iters = math.ceil(len(dataset['tokens']) / self.batch_sz)
			# Get the primary classifier
			iterator = tqdm(iterator, total= total_iters, desc="Classifier Train Iterator")
			for idx, samples in enumerate(iterator):
				if (idx + 1) % self.grad_accum_factor == 0:
					# We want to take a gradient step after accumulating gradients
					torch.nn.utils.clip_grad_norm_(prim_classf.parameters(), max_grad_norm)
					optimizer.step()
					if lr_scheduler is not None:
						lr_scheduler.step()
					optimizer.zero_grad()
				output_dict = prim_classf(*samples)
				total_loss = output_dict['loss'] / self.grad_accum_factor  # Account for fact that we are accumulating gradients
				total_loss.backward()
			# We want to evaluate the classifier
			train_metrics = self.get_metrics(reset=True)
			dev_metrics  = self.evaluate_classifier(set_='dev')
			test_metrics = self.evaluate_classifier(set_='test')
			# Report the metrics
			for k, v in train_metrics.items():
				to_show = k, v, dev_metrics[k], test_metrics[k]
				print_out = "[{}] | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(*to_show)
				print(print_out)
			self.perfs['train'].append((train_metrics['f1'], train_metrics['accuracy']))
			self.perfs['dev'].append((dev_metrics['f1'], dev_metrics['accuracy']))
			self.perfs['test'].append((test_metrics['f1'], test_metrics['accuracy']))
			metric_idx = 0 if metric == 'f1' else 1
			if dev_metrics[metric] >= self.perfs['dev'][best_iter][metric_idx]:
				best_iter = iter_
				iters_since_improvement = 0
			else:
				iters_since_improvement += 1
				if iters_since_improvement >= patience:
					print('Breaking because we have no improvement in {} epochs'.format(patience))
					break
		best_f1, best_acc = self.perfs['test'][best_iter]
		return best_f1, best_acc, self.perfs, self.perfs['dev'][best_iter]


	'''
		Data processing related functionality
	'''
	# Get dataset corresponding to a particular split of the data
	def get_dataset(self, split):
		if self.data_splits[split] is not None:
			return self.data_splits[split]

		inputdict = {split: self.file_for_split[split]}
		dataset = self.setup_datasets(
										inputdict, self.model_name, self.max_seq_len,
										label_vocab=self.label_vocab
									)[split]
		self.data_splits[split] = dataset
		if split == 'train':
			# Initialize the label vocab if this is the first time we are accessing the training dataset
			self.label_vocab = dataset['vocab']
		return dataset

	# Load / Read a dataset from file.
	# Files are expected to be json format. This deals with classification datasets
	def setup_datasets(self, dataset_files, model_name, max_seq_len, label_vocab=None, lazy=False):
		# Instantiate dataset reader
		datasets = defaultdict(dict)
		indexers = {'tokens': PretrainedTransformerIndexer(model_name, do_lowercase=False)}
		tokenizer = PretrainedTransformerTokenizer(model_name, do_lowercase=False, start_tokens=["<s>"], end_tokens=["</s>"])
		dataset_reader = TextClassificationJsonReaderWithSampling(
							token_indexers=indexers, tokenizer=tokenizer,
							max_sequence_length=max_seq_len, lazy=lazy
						)
		# Read from the dataset
		pretrain_vocab = tokenizer._tokenizer.encoder
		for idx_, fname in dataset_files.items():
			all_samples = dataset_reader._read(fname)
			all_sentences, all_instances = [], []
			lens = []
			for instance in all_samples:
				tokens = instance.fields['tokens']
				tokens.index(pretrain_vocab)
				sentence = tokens.as_tensor(tokens.get_padding_lengths())['tokens']
				all_sentences.append(sentence)
				all_instances.append(instance)
				lens.append(sentence.shape[0])

			if label_vocab is not None:
				vocab = label_vocab
			else:
				vocab = Vocabulary.from_instances(all_instances)
			all_labels = []
			for instance in all_instances:
				label = instance.fields['label']
				label.index(vocab)
				this_label = label.as_tensor(label.get_padding_lengths())
				all_labels.append(this_label)
			datasets[idx_] = {
								'tokens': all_sentences,
								'labels': np.array(all_labels),
								'pad_idx': tokenizer._tokenizer.pad_token_id,
								'vocab': vocab
							}
		return datasets

	# Get nsamples from a particular split of the task dataset
	def get_data_samples(self, split, nsamples):
		dataset = self.get_dataset(split)
		num_egs = len(dataset['tokens'])
		idxs = np.random.choice(num_egs, size=nsamples, replace=(num_egs < nsamples))
		sentences, labels = [dataset['tokens'][i] for i in idxs], dataset['labels'][idxs]
		sentences = collate(sentences, dataset['pad_idx'])
		sentences = sentences.to(self.base_lm_model.device)
		labels = torch.IntTensor(labels).to(sentences.device)
		return sentences, labels

	# Dataset iterator.
	# Yields batches of specified dataset
	def dataset_iterator(self, dataset, shuffle=False, batchsz=-1):
		if batchsz < 0:
			# Use the prescribed default batch size
			batchsz = self.batch_sz
		total_egs = len(dataset['tokens'])
		num_batches = math.ceil(total_egs / batchsz)
		if shuffle:
			idxs = np.random.permutation(total_egs)
		else:
			idxs = list(range(total_egs))
		for i in range(num_batches):
			this_idxs = idxs[(i * batchsz): ((i + 1) * batchsz)]
			sentences = [dataset['tokens'][id_] for id_ in this_idxs]
			labels = dataset['labels'][this_idxs]
			sentences = collate(sentences, dataset['pad_idx'])
			sentences = sentences.to(self.base_lm_model.device)
			labels = torch.IntTensor(labels).to(self.base_lm_model.device)
			yield sentences, labels
			# Clean up after yielding
			del sentences
			del labels

