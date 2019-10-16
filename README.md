# What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis

## Description
This repository includes the source code of the paper "What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis". Please cite our paper when you use this program! 😍

## Model overview
![](https://i.loli.net/2019/10/16/24uzEvdC8OFkSnX.png)

## Requirements
python3

```
pip install -r requirements.txt
```

## Datasets
Our code can run on four datasets.

### Duplicate question identification datasets
1. Quora Question Pairs
2. CQADupStack

### Natural language inference datasets
1. SNLI
2. MultiNLI

## Data format

## Preprocess the data
After downloading the dataset, you can preprocess the data.

### Preprocess the data for BERT
```
cd scripts/preprocessing
python process_quora_bert.py
python preprocess_snli_bert.py
python process_mnli_bert.py
python preprocess_cqadup_bert.py
```

### Preprocess the data for ELMo
```
cd scripts/preprocessing
python process_quora.py
python preprocess_snli.py
python preprocess_mnli.py
```

## Train
### BERT service
If you want to train models with BERT word embedding, please use the [bert-as-service](https://github.com/hanxiao/bert-as-service), then run the following scripts.

### Train all models
```
sh -x run.sh
```

### Train with BERT
```
python bert_quora.py >> log/quora/quora_bert.log
python bert_snli.py >> log/snli/snli_bert.log
python bert_mnli.py >> log/mnli/mnli_bert.log
python bert_cqadup.py >> log/cqadup/cqadup_bert.log
```

### Train with ElMo
```
python train_quora_elmo.py >> log/quora/quora_elmo.log
python train_mnli_elmo.py >> log/snli/snli_elmo.log
python train_snli_elmo.py >> log/mnli/mnli_elmo.log
```

## Test
After each training epoch, we will test on the valid/test set. You can select the train/valid/test process according to your needs.

## Reporting issues
Please let me know, if you encounter any problems.

The contact email is rzhangpku@pku.edu.cn


