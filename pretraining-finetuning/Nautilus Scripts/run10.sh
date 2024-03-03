#!/bin/bash


python3 input_visit_time/pretrain/run_pretraining.py -c bert_dataset_train.pkl -vl bert_dataset_valid.pkl -config input_visit_time/pretrain/config.json -v behrt_features.types --checkpoint_dir 'input_visit_time/pretrain/results/' --epochs 500
