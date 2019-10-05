#!/usr/bin/env bash
python bert_quora.py > log/quora/quora_bert4.log
python bert_snli.py >> log/snli/snli_bert4.log
python bert_mnli.py >> log/mnli/mnli_bert4.log