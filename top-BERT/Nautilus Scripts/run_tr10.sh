python3 input_visit_time/tribert/predict_all.py -c FINETUNING_LIST_YR1_TRAIN.pkl -t FINETUNING_LIST_YR1_TEST.pkl -v FINETUNING_LIST_YR1_VALID.pkl  --checkpoint_dir 'input_visit_time/tribert/results' -config input_visit_time/tribert/config.json -vocab behrt_features.types --epochs 500 --clip_grads


python3 ig1.py  -t FINETUNING_LIST_YR1_TEST.pkl -p 'pytorch_model.bin' -config config.json -vocab behrt_features.types -d input_vist_time


cd aim1/input_visit_time/tribert

python3 test_auc.py  -t FINETUNING_LIST_YR1_TEST.pkl -p 'pytorch_model.bin' -config config.json -vocab behrt_features.types -d input_vist_time


python3 test_auc.py  -t FINETUNING_LIST_YR1_TEST.pkl -p 'pytorch_model.bin' -config config.json -vocab behrt_features.types -d finetune1_input_vist_time

python3 test_auc.py  -t FINETUNING_LIST_YR1_TEST.pkl -p 'pytorch_model.bin' -config config.json -vocab behrt_features.types -d finetune2_input_vist_time