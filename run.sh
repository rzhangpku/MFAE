#!/usr/bin/env bash
## Train with BERT
python bert_quora.py >> log/quora/quora_bert.log
python bert_cqadup.py >> log/cqadup/cqadup_bert.log
python bert_snli.py >> log/snli/snli_bert.log
python bert_mnli.py >> log/mnli/mnli_bert.log
## Train with ElMo
python train_quora_elmo.py >> log/quora/quora_elmo.log
python train_snli_elmo.py >> log/snli/snli_elmo.log
python train_mnli_elmo.py >> log/mnli/mnli_elmo.log
