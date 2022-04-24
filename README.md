# Should we be Pre-Training ? Exploring End-Task Aware Training In Lieu of Continued Pre-training

This repository contains the source code for the paper [Should we be Pre-Training ? Exploring End-Task Aware Training In Lieu of Continued Pre-training](https://openreview.net/forum?id=2bO2x8NAIMB), by Lucio M Dery, Paul Michel, Ameet Talwalkar and Graham Neubig (ICLR 2022).

---

<p align="center"> 
    <img src="https://github.com/ldery/TARTAN/blob/main/eatmt.png" width="800">
</p>

## Links

1. [Paper](https://openreview.net/forum?id=2bO2x8NAIMB)
2. Bibtext :
```
@inproceedings{
dery2022should,
title={Should We Be Pre-training? An Argument for End-task Aware Training as an Alternative},
author={Lucio M. Dery and Paul Michel and Ameet Talwalkar and Graham Neubig},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=2bO2x8NAIMB}
}
```

## Installation Instructions
This repo builds off the Don't Stop Pre-training paper repo [here](https://github.com/allenai/dont-stop-pretraining). 
Please follow their installation instructions. Repeated here for ease:

```bash
conda env create -f environment.yml
conda activate domains
```

## Running
Our experiments were run on A6000 and A100 gpus which have > 40G gpu memory. To ensure that batches fit into memory, consider modifying the following variables

```bash
--classf_iter_batchsz   # Batch Size for primary task. Effective batchsize is  (classf_iter_batchsz * gradient_accumulation_steps)
--per_gpu_train_batch_size  # Batch Size for auxiliary tasks. Effective batchsize is  (per_gpu_train_batch_size * gradient_accumulation_steps)
--gradient_accumulation_steps # Number of timesteps to accumulate gradient over
```

### To obtain results on sample datasets

#### Baseline 
We used the TAPT baseline from the Don't Stop Pre-training paper. To reproduce this baseline, please follow the instructions in their repo [here](https://github.com/allenai/dont-stop-pretraining) - to download and run their pre-trained models.


#### Ours 
###### MT-TARTAN
./run_mt_multiple.sh {task} {outdir} {gpuid} {startseed} {endseed}

###### META-TARTAN
./run_meta_multiple.sh {task} {outdir} {gpuid} {startseed} {endseed}