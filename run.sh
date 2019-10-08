#!/usr/bin/env bash
python bert_quora.py > log/quora/quora_bert.log
python bert_cqadup.py > log/cqadup/cqadup_bert.log
python bert_snli.py >> log/snli/snli_bert.log
python bert_mnli.py >> log/mnli/mnli_bert.log