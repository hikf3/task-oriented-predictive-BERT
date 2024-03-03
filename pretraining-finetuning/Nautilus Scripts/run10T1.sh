#!/bin/bash


python3 input_visit_time/pretrain/finetuning.py -c FINETUNING_LIST_YR1_TRAIN.pkl -t FINETUNING_LIST_YR1_TEST.pkl -v FINETUNING_LIST_YR1_VALID.pkl -p 'input_visit_time/pretrain/pytorch_model.bin' --checkpoint_dir 'input_visit_time/pretrain/t1' -config input_visit_time/pretrain/config.json -vocab behrt_features.types --epochs 300 --clip_grads


