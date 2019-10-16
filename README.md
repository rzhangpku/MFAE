# What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis

## Description
This repository includes the source code of the paper "What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis". Please cite our paper when you use this program! 😍

## Model overview
![Overview of MFAE](https://i.loli.net/2019/10/12/eSbyd2jfRtZnIQA.png)

## Requirements
python

pip install requirements.txt

## Datasets
Our code can run on four datasets.

Duplicate question identification datasets:
1. Quora Question Pairs
2. CQADupStack

Natural language inference datasets:
1. SNLI
2. MultiNLI

## Data format

## Preprocess the data
After download the dataset, you can run process_quora_bert.py, process_quora.py(elmo), process_mnli_bert.py et. in scripts/preprocessing to preprocess the data.

## Train
For convenience, you can simply run run.sh to train the model based on BERT service (https://github.com/hanxiao/bert-as-service) on every dataset one by one.
You can also select the specific file(bert_quora.py, bert_cqadup.py,
bert_mnli.py, bert_snli.py) to train AMAE on BERT or (train_quora_elmo.py, train_mnli_elmo.py, train_snli_elmo.py) on Elmo.
## Test

After each training epoch, we will test on the valid/test set. You can select the train/valid/test process according to your needs.

## Reporting issues
Please let me know, if you encounter any problems.

The contact email is rzhangpku@pku.edu.cn
