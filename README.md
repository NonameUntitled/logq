# Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval

This repository is a fork of the original gSASRec-pytorch implementation. It is dedicated to our paper: "Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval"

### Overview
This repo maintains the codebase of gSASRec for reproducibility. Additionally we provided configuration files and launch scripts for training SASRec with different regimes (in-batch/uniform/full softmax/etc).

What’s included:
- Original gSASRec code
- Additional config files for running experiments as described in our paper (for ML1M, Steam and Gowalla dataset)
- Launch and training scripts to reproduce our results

## Getting Started

To train or evaluate the model with our configurations you should run following commands:


## Training Example

### SASRec

Implementation of original SASRec which utilizes BPR loss:
```
python train_sasrec.py --config=configs/ml1m_sasrec.py
```

### gSASRec
Implementaion of "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling" paper:
```
python train_gsasrec.py --config=configs/ml1m_gsasrec.py
```

### SASRec with full softmax
Implemenation which utilizes cross entropy loss over the whole set of items:
```
python train_full_softmax.py --config=configs/ml1m_other.py 
```

### SASRec with uniformly sampled negatives
Implementation which uniformly subsamples random items from corpus and computes sampled softmax loss over them:
```
python train_uniform.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives
Implementation which subsamples negatives for each interaction from the current mini-batch:
```
python train_in_batch.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives with original log-q correction
Implementation which subsamples negatives for each interaction from the current mini-batch and applies original log-q correction:
```
python train_in_batch_log_old.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives with our implementation of log-q correction
Implementation which subsamples negatives for each interaction from the current mini-batch and applies our implementation of log-q correction:
```
python train_in_batch_log_new.py --config=configs/ml1m_other.py 
```

### SASRec with mixed negative sampling (MNS)
Implementation which subsamples negatives for each interaction both uniformly from corpus and from the current mini-batch (in even parts):
```
python train_mns.py --config=configs/ml1m_other.py 
```

### SASRec with mixed negative sampling (MNS) with original log-q correction
Implementation which subsamples negatives for each interaction both uniformly from corpus and from the current mini-batch (in even parts) and applies original log-q correction:
```
python train_mns_logq_old.py --config=configs/ml1m_other.py 
```

### SASRec with mixed negative sampling (MNS) with our implementation of log-q correction
Implementation which subsamples negatives for each interaction both uniformly from corpus and from the current mini-batch (in even parts) and applies our implementation of log-q correction:
```
python train_mns_logq_new.py --config=configs/ml1m_other.py 
```

## Other datasets

To use different dataset just replace 'ml1m' with 'steam' or 'gowalla' in config filename.
There are also configs dedicated to time-split (ones with '_time' preffix after the dataset name).


## Evaluation Example

For every model evaluation you need to use the same file but different configs.

### Example

For evaluating SASRec with full softmax on Steam dataset run:
```
python evaluate.py --config=configs/steam_sasrec_other.py --checkpoint=your_checkpoint.pt
```
Replace `your_checkpoint.pt` with the appropriate file that is produced after the end of the training stage.


### About gSASRec

gSASRec is a sequential recommendation model based on SASRec, enhanced with improved negative sampling and a generalized BCE loss (gBCE). 
The model architecture is based on a Transformer decoder, with differences primarily in the training procedure rather than the model structure itself.

For more details on the original model, see the RecSys '23 paper:
"gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling" by Aleksandr Petrov and Craig Macdonald.
