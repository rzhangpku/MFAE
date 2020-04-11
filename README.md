# What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis
# SDM 2020

## Description
This repository includes the source code of the paper "What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis". Please cite our paper when you use this program! 😍 This paper has been accepted to the conference "SIAM International Conference on Data Mining (SDM20)".

```
@inproceedings{zhang2020what,
  title={What Do Questions Exactly Ask? MFAE: Duplicate Question Identification with Multi-Fusion Asking Emphasis},
  author={Zhang, Rong and Zhou, Qifei and Wu, Bo and Li, Weiping and Mo, Tong},
  booktitle={Proceedings of the 2020 SIAM International Conference on Data Mining},
  year={2020},
  organization={SIAM}
}
```

## Model overview
![](https://i.loli.net/2019/10/16/24uzEvdC8OFkSnX.png)

## Requirements
python3

```
pip install -r requirements.txt
```

## Datasets
Our code can run on four datasets.

### Duplicate Question Identification Datasets (DQI)
* Quora Question Pairs
* CQADupStack

### Natural Language Inference Datasets (NLI)
* SNLI
* MultiNLI

## Data Preprocessing
After the datasets have been downloaded, you can preprocess the data.

### Preprocess the data by BERT
```
cd scripts/preprocessing
python process_quora_bert.py
python preprocess_cqadup_bert.py
python preprocess_snli_bert.py
python process_mnli_bert.py
```

### Preprocess the data by ELMo
```
cd scripts/preprocessing
python process_quora.py
python preprocess_snli.py
python preprocess_mnli.py
```

## Training
### BERT as service
If you want to train models with BERT word embedding, please use the [bert-as-service](https://github.com/hanxiao/bert-as-service), and then run the following scripts.

### Train all models
```
sh -x run.sh
```

### Train with BERT
```
python bert_quora.py >> log/quora/quora_bert.log
python bert_cqadup.py >> log/cqadup/cqadup_bert.log
python bert_snli.py >> log/snli/snli_bert.log
python bert_mnli.py >> log/mnli/mnli_bert.log
```

### Train with ELMo
```
python train_quora_elmo.py >> log/quora/quora_elmo.log
python train_snli_elmo.py >> log/snli/snli_elmo.log
python train_mnli_elmo.py >> log/mnli/mnli_elmo.log
```

## Testing
After the models have been trained, you can test the models.

### Test the models with BERT backbone

```
python test_bert_quora.py
python test_bert_cqadup.py
python test_bert_snli.py
python test_bert_mnli.py
```

### Test the models with ELMo backbone

```
python test_elmo_quora.py
python test_elmo_snli.py
python test_elmo_mnli.py
```

## Report issues
Please let us know, if you encounter any problems.

The contact email is rzhangpku@pku.edu.cn


