# Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval

This repository is a fork of the original gSASRec-pytorch implementation. We provide additional configuration files and launch scripts for training SASRec under various regimes (in-batch, uniform, full softmax, etc.).

It is dedicated to our paper:
"Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval"

## Getting Started

To train or evaluate the model with our configurations, use the following commands:

## Training Example

### SASRec (Original)

Implementation of theoriginal SASRec with BPR loss:
```
python train_sasrec.py --config=configs/ml1m_sasrec.py
```

### gSASRec
Implementaion of the model from "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling":
```
python train_gsasrec.py --config=configs/ml1m_gsasrec.py
```

### SASRec with full softmax
Implemenation which utilizes cross-entropy loss over the entire item set:
```
python train_full_softmax.py --config=configs/ml1m_other.py 
```

### SASRec with uniformly sampled negatives
Implementation which uniformly subsamples random items from the corpus and computes sampled softmax loss:
```
python train_uniform.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives
Implementation which subsamples negatives for each positive interaction from the current mini-batch:
```
python train_in_batch.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives with original log-q correction
Implementation which subsamples negatives for each positive interaction from the current mini-batch and applies **the original logQ correction**:
```
python train_in_batch_logq_old.py --config=configs/ml1m_other.py 
```

### SASRec with in-batch negatives with our implementation of log-q correction
Implementation which subsamples negatives for each positive interaction from the current mini-batch and applies **our implementation of logQ correction**:
```
python train_in_batch_logq_new.py --config=configs/ml1m_other.py 
```

### SASRec with Mixed Negative Sampling (MNS)
Implementation which subsamples negatives both uniformly from the corpus and from the current mini-batch (even split):
```
python train_mns.py --config=configs/ml1m_other.py 
```

### SASRec with mixed negative sampling (MNS) with original log-q correction
Implementation which leverages mixed negative sampling with **the original logQ correction**:
```
python train_mns_logq_old.py --config=configs/ml1m_other.py 
```

### SASRec with mixed negative sampling (MNS) with our implementation of log-q correction
Implementation which leverages mixed negative sampling with **our implementation of log-q correction**:
```
python train_mns_logq_new.py --config=configs/ml1m_other.py 
```

## Using Other Datasets

To use a different dataset, replace `ml1m` in the config filename with `steam` or `gowalla`.
Configs with a `_time` suffix after the dataset name are dedicated to time-based splits.

## Evaluation Example

To evaluate any model, use the same script with the appropriate config and checkpoint:

### Example

To evaluate SASRec with full softmax on the Steam dataset:
```
python evaluate.py --config=configs/steam_sasrec_other.py --checkpoint=your_checkpoint.pt
```
Replace `your_checkpoint.pt` with the checkpoint produced after training.
