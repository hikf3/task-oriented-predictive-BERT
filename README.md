# task-oriented-predictive-BERT
Shifting from the usual pretraining-finetuning model, we introduced an end-to-end training and evaluation method called task-oriented-predictive (top)-BERT. In top-BERT, we utilized the sequential input structure, embedding layer, and encoder stacks inherent to BERT to train and evaluate three tasks simultaneously: the conventional Masked Language Model (MLM), a binary classification for prolonged hospital stay (1 if the length of stay >7, else 0), and a multilabel sequence classification for the four complications mentioned above. We aggregated the loss of the three tasks, which was backpropagated throughout the entire network, leading to improved learning of our model in a limited cohort sample size. 

![image](https://github.com/hikf3/task-oriented-predictive-BERT/assets/72622960/5081fba6-0218-4552-8701-76b4576d4dbe)
![top-BERT](https://github.com/hikf3/task-oriented-predictive-BERT/assets/72622960/8ac1b57f-5347-4a89-a0ab-a42b98ad6db5)


![top-BERT_implementation](https://github.com/hikf3/task-oriented-predictive-BERT/assets/72622960/744d28cb-6717-4d74-8b80-003040bfe465)
