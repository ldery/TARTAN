
CITATION = {
	'citation_intent': 'datasets/citation_intent/train.jsonl',
	'train': 'datasets/citation_intent/train.jsonl',
	'dev': 'datasets/citation_intent/dev.jsonl',
	'test': 'datasets/citation_intent/test.jsonl',
}

CHEMPROT = {
	'chemprot': 'datasets/chemprot/train.jsonl',
	'train': 'datasets/chemprot/train.jsonl',
	'dev': 'datasets/chemprot/dev.jsonl',
	'test': 'datasets/chemprot/test.jsonl',
}

SCIIE = {
	'sciie': 'datasets/sciie/train.jsonl',
	'train': 'datasets/sciie/train.jsonl',
	'dev': 'datasets/sciie/dev.jsonl',
	'test': 'datasets/sciie/test.jsonl'
}

HYPERPARTISAN = {
	'hyperpartisan': 'datasets/hyperpartisan/train.jsonl',
	'train': 'dsp/datasets/hyperpartisan/train.jsonl',
	'dev': 'dsp/datasets/hyperpartisan/dev.jsonl',
	'test': 'dsp/datasets/hyperpartisan/test.jsonl'
}


def get_auxtask_files(task_name):
	if task_name == 'citation_intent':
		return CITATION
	elif task_name == 'chemprot':
		return CHEMPROT
	elif task_name == 'sciie':
		return SCIIE
	elif task_name == 'hyperpartisan':
		return HYPERPARTISAN
	else:
		raise ValueError