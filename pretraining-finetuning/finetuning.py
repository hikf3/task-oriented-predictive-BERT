import torch
import torch.nn as nn
#from optimizers import NoamOptimizer
from modeling import BertModel, BertConfig, BertForSequenceClassification, BertForPreTraining
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler,BatchSampler
#from dataloader import InputFeatures, BERTdataEHR, BERTdataEHRloader
from log_utils import make_run_name, make_logger, make_checkpoint_dir
import pickle as pkl
import time
import logging
import argparse
from tqdm import tqdm, trange
import sys
import pandas as pd
import random
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import os
from os.path import join
from datetime import datetime
from termcolor import colored
import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

experiment='finetune-yr1-10-input_visit_time'

wandb.login(key='8775b926c56fc57a316bb306aa7e0dc876f54821')

run=wandb.init(
    project="ehr_bert-"+experiment,
    config={
        "model":"Finetune-Year1",
        "Optimizer": "AdamW",
        "loss_fn":"wBinaryCrossEntropy",
        "epochs": "250",
        "INPUT_DIM":"25k",
        "HIDDEN_DIM":"128",        
    }
)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Check GPU status
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logging.info('CUDA STATUS: %s', use_cuda)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s).")
    #logging.info(f"Found {device_count} CUDA device(s).")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
        #logging.info(f"Device {i}: {device_name}")
else:
    print("CUDA is not available.")
    #logging.info("CUDA is not available.")


SAVE_FORMAT = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

LOG_FORMAT = (
    "Epoch: {epoch:>3} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "Examples/second: {per_second:<.1} "
    "Train Loss: {train_loss:<.6} "
    "Valid Loss: {val_loss:<.6} "
    "Train Metrics: {train_metrics} "
    "Val Metrics: {val_metrics} "
    "Learning rate: {current_lr:<.4} "
)

RUN_NAME_FORMAT = (
    "BERT-"
    "{phase}-"
    "num_hidden_layers={num_hidden_layers}-"
    "hidden_size={hidden_size}-"
    "num_attention_heads={num_attention_heads}-"
    "{timestamp}"
)



### Below are key functions for  Data preparation,formatting input data into features, and model definition 

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  batches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               timetonext_ids,
               enc_age_ids,
               visit_segment_ids,
               label_id,
               patient_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.timetonext_ids = timetonext_ids
    self.enc_age_ids = enc_age_ids
    self.segment_ids = segment_ids
    self.visit_segment_ids = visit_segment_ids
    self.label_id = label_id
    self.patient_id = patient_id
    self.is_real_example = is_real_example

    

    
def convert_EHRexamples_to_features(examples,max_seq_length):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_singleEHR_example(ex_index, example, max_seq_length)
        features.append(feature)
    return features

### This is the EHR version

def convert_singleEHR_example(ex_index, example, max_seq_length):
    
    input_ids = example[4]
    segment_ids = example[7]
    patient_id = example[0]
    timetonext_ids = example[3]
    enc_age_ids = example[5]
    visit_segment_ids = example[6]
    label_id = example[1]


    # insert [CLS] and adjust the other lists
    input_ids.insert(0,2)
    segment_ids.insert(0,segment_ids[0])
    timetonext_ids.insert(0,timetonext_ids[0])
    enc_age_ids.insert(0,enc_age_ids[0])
    visit_segment_ids.insert(0,visit_segment_ids[0])
 
  # Left Truncate longer sequence
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        input_mask = [1] * max_seq_length
        timetonext_ids = timetonext_ids[-max_seq_length:]
        enc_age_ids = enc_age_ids[-max_seq_length:]
        visit_segment_ids = visit_segment_ids[-max_seq_length:]
        segment_ids = segment_ids[-max_seq_length:]
    else:
        # No truncation needed, create input_mask
        input_mask = [1] * len(input_ids)
        # Zero-pad the sequences to match max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            timetonext_ids.append(0)
            enc_age_ids.append(0)
            visit_segment_ids.append(0)
            segment_ids.append(0)  

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(timetonext_ids) == max_seq_length
    assert len(enc_age_ids) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(visit_segment_ids) == max_seq_length

  
    #feature =[input_ids,input_mask,segment_ids,label_id,patient_id,True]
    #print('######Features', feature)
    
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        timetonext_ids=timetonext_ids,
        enc_age_ids=enc_age_ids,
        visit_segment_ids = visit_segment_ids,
        segment_ids=segment_ids,
        label_id=label_id,
        patient_id=patient_id,
        is_real_example=True
    )
    
    return feature

## BERTdataEHR creating a PyTorch 'Dataset' object from the features  that can be used to feed the input data to a PyTorch model

class BERTdataEHR(Dataset):
    def __init__(self, Features):
        self.data = Features

    def __getitem__(self, idx, seeDescription=False):
        sample = self.data[idx]
        # Create a dictionary to store the features
        feature_dict = {
            "input_ids": sample.input_ids,
            "input_mask": sample.input_mask,
            "timetonext_ids": sample.timetonext_ids,
            "enc_age_ids": sample.enc_age_ids,
            "visit_segment_ids": sample.visit_segment_ids,
            "segment_ids": sample.segment_ids,
            "label_id": sample.label_id,
            "patient_id": sample.patient_id,
            "is_real_example": sample.is_real_example
        }
        return feature_dict, torch.tensor(sample.label_id, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

def my_collate(batch):
    all_input_ids = []
    all_input_mask = []
    all_timetonext_ids =[]
    all_enc_age_ids =[]
    all_visit_segment_ids =[]
    all_segment_ids = []
    all_label_ids = []
    all_patient_ids = []

    all_labels =[]

    for feature, labels in batch:
        all_input_ids.append(feature["input_ids"])
        all_input_mask.append(feature["input_mask"])
        all_timetonext_ids.append(feature["timetonext_ids"])
        all_enc_age_ids.append(feature["enc_age_ids"])
        all_visit_segment_ids.append(feature["visit_segment_ids"])
        all_segment_ids.append(feature["segment_ids"])
        all_label_ids.append(feature["label_id"]) # Convert 0-dimensional tensor to a scalar
        all_patient_ids.append(feature["patient_id"])
        all_labels.append(labels)

    return {
        "input_ids": torch.tensor(all_input_ids),
        "input_mask": torch.tensor(all_input_mask),
        "timetonext_ids": torch.tensor(all_timetonext_ids),
        "enc_age_ids": torch.tensor(all_enc_age_ids),
        "visit_segment_ids": torch.tensor(all_visit_segment_ids),
        "segment_ids": torch.tensor(all_segment_ids),
        "label_id": torch.tensor(all_label_ids),
        "patient_id": torch.tensor(all_patient_ids),
        "labels": torch.stack(all_labels)
    }
class BERTdataEHRloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        # Calculate weights based on the inverse of the class frequency
        all_zero_count = sum(1 for item in dataset if not item[1].bool().any())
        at_least_one_count = len(dataset) - all_zero_count
        
        # Calculate weights
        weight_all_zero = 1.0 / all_zero_count if all_zero_count > 0 else 1.0
        weight_at_least_one = 1.0 / at_least_one_count if at_least_one_count > 0 else 1.0
        # Assign weights to the sampler
        weights = [weight_all_zero if not item[1].bool().any() else weight_at_least_one for item in dataset]
        # Create class indices based on weights
        class_indices = [
            idx for idx, item in enumerate(dataset) if item[1].bool().any() and torch.rand(1).item() < weights[idx]
        ]
        
        #weights = [class_weights[item[1].bool()].sum().item() for item in dataset]
        sampler = WeightedRandomSampler(weights, len(dataset))
        #batch_sampler = BatchSampler(class_indices, batch_size=batch_size, drop_last=False)
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=sampler, batch_sampler=None, num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn


## Defining the model that can produce multiple class output from BERT

class EHRBertMultiOutputClassifier(nn.Module):
    
    def __init__(self, config, model, num_labels=4):
        super(EHRBertMultiOutputClassifier, self).__init__()
        self.num_labels = num_labels
        self.pretrainedbert = model
        #Add a dropout 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Add a classification layer on top of the BERT model
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Add a Sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids,  visit_ids=None, timetonext_ids=None,  attention_mask=None, labels=None):
        outputs = self.pretrainedbert.bert(input_ids,  visit_ids,  timetonext_ids, attention_mask)

        out= outputs[1] #pooled output
        logits =self.classifier(self.dropout(out))
        return logits

## Defining the loss function

class BertFineTuningCriterion(nn.Module):
    def __init__(self, num_labels, loss_weights):
        super(BertFineTuningCriterion, self).__init__()

        self.device = torch.device("cuda:0")
        self.class_weights = loss_weights.to(self.device)
        # Loss functions
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.class_weights, reduction = 'mean')
        self.num_labels = num_labels
        
    def forward(self, logits, labels):
        loss = self.loss_fn(logits, labels.float())
        
        return loss

def calculate_metrics(output, labels, num_labels, epoch, epoch_loss, elapsed):
    # Initialize arrays to store AUC scores for each label
    auc_scores = []
    
    # Initialize arrays to store precision, recall, and F1 scores for each label and threshold
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Convert labels and logits to numpy arrays
    y_true = labels.cpu().numpy()
    #print('##### Y Labels',y_true)
    #print('##### Output', output)

    sigmoid=nn.Sigmoid()
    #y_scores = F.softmax(output.cpu(), dim=-1).detach().numpy()
    y_logits = output.cpu().detach().numpy()
    # Apply nn.Sigmoid to logits
    y_scores = sigmoid(torch.from_numpy(y_logits)).numpy()
    #print('##### Scores after sigmoid', y_scores)

    #y_pred_labels = (y_scores > 0.5).astype(int)

    #print('##### Predicted labels', y_pred_labels)
    #thresholds= [0.1, 0.20, 0.20, 0.05]
    thresholds= [0.5, 0.5, 0.5, 0.5]
    for label_idx in range(num_labels):
        y_true_label = y_true[:, label_idx]
        y_scores_label = y_scores[:, label_idx]
        
        # Calculate AUC for the current label
        auc_label = roc_auc_score(y_true_label, y_scores_label)
        auc_scores.append(auc_label)

        # Iterate over different threshold values
        threshold = thresholds[label_idx]
        
        # Calculate predicted labels for the current threshold
        y_pred_labels = (y_scores_label > threshold).astype(int)

        # Calculate precision, recall, and F1 for the current label and threshold
        precision_label = precision_score(y_true_label, y_pred_labels, zero_division=0)
        recall_label = recall_score(y_true_label, y_pred_labels, zero_division=0)
        f1_label = f1_score(y_true_label, y_pred_labels, zero_division=0)

        # Store precision, recall, and F1 for the current label and threshold
        precision_scores.append(precision_label)
        recall_scores.append(recall_label)
        f1_scores.append(f1_label)

    print('### auc', auc_scores)
    print('### precision', precision_scores)
    print('### recall', recall_scores)
    print('### f1', f1_scores)
    
    
    # Calculate macro-average AUC, precision, recall, and F1
    macro_auc = np.mean(auc_scores)
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)

    # Calculate micro-average AUC, precision, recall, and F1
    micro_auc = roc_auc_score(y_true.ravel(), y_scores.ravel())
    micro_precision = precision_score(y_true.ravel(), (y_scores.ravel() > 0.5).astype(int), average='weighted', zero_division=0)
    micro_recall = recall_score(y_true.ravel(), (y_scores.ravel() > 0.5).astype(int), average='weighted', zero_division=0)
    micro_f1 = f1_score(y_true.ravel(), (y_scores.ravel() > 0.5).astype(int), average='weighted', zero_division=0)
    print('### micro auc', micro_auc)
    print('### micro precision', micro_precision)
    print('### micro recall', micro_recall)
    print('### micro f1', micro_f1)
    metric_save = {
            'epoch': epoch,
            'loss': epoch_loss,
            'AUC': auc_scores,
            'macro_AUC': macro_auc,
            'micro_AUC': micro_auc,
            'precision': precision_scores,
            'macro_precision': macro_precision,
            'micro_precision': micro_precision,
            'recall': recall_scores,
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'f1': f1_scores,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'time cost': elapsed
        }
           
    return metric_save

def calculate_auc(output, labels, num_labels):
    # Initialize arrays to store AUC scores for each label
    auc_scores = []
    
    # Initialize variables for micro-average calculation
    y_true = labels.cpu().numpy()
    y_logits = output.cpu().detach().numpy()
    sigmoid=nn.Sigmoid()
    
    # Apply nn.Sigmoid to logits
    y_scores = sigmoid(torch.from_numpy(y_logits)).numpy() 

    # Calculate AUC for each label separately
    for label_idx in range(num_labels):
        y_true_label = y_true[:, label_idx]
        y_score_label = y_scores[:, label_idx]
        
        # Calculate AUC for the current label
        auc_label = roc_auc_score(y_true_label, y_score_label)
        auc_scores.append(auc_label)

    # Calculate macro-average AUC
    macro_auc = np.mean(auc_scores)

    # Calculate micro-average AUC
    micro_auc = roc_auc_score(y_true.ravel(), y_scores.ravel())
    
    return macro_auc, micro_auc


class BertTrainer:
    def __init__(self, model, loss_model, train_dataloader,valid_dataloader,test_dataloader,
                 with_cuda, optimizer, scheduler, clip_grads,
                 logger, checkpoint_dir, print_every, save_every):
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        self.model = model.to(self.device)
        self.loss_model = loss_model
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grads = clip_grads

        self.logger = logger
        self.checkpoint_dir = checkpoint_dir

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_output_path = None

        self.flt_typ = torch.cuda.FloatTensor
        self.lnt_typ = torch.cuda.LongTensor


    def run_epoch(self, batch_dataloader, mode='train'):

        epoch_loss = 0
        epoch_count = 0
        all_logits =[]
        all_labels = []
        
        batch_iter = tqdm(
            batch_dataloader,
            desc="Iteration",
            disable=False,
            total=len(batch_dataloader))

        for batch_count, batch in enumerate(batch_iter):
                #0. batch_data will be sent into the device(GPU or cpu)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                #1. forward the next_sentence_prediction and masked_lm model
                logits = self.model(input_ids=batch['input_ids'], 
                                    visit_ids = batch['visit_segment_ids'],
                                    timetonext_ids =batch['timetonext_ids'], 
                                    attention_mask=batch['input_mask'])
                # Check the dimensions of logits
                #print(logits.shape)
                labels = batch['label_id']
                #print('####Labels for batch',labels)
                
                # Appending the tensors
                #y_real.extend(labels.cpu().data.view(-1).numpy())
                #y_hat.append(logits.cpu().data.view(-1).numpy())
                all_labels.append(labels)
                all_logits.append(logits)
    
                #2. calculate loss
                loss = self.loss_model(logits.type(self.flt_typ), labels.type(self.flt_typ) )
                

                batch_losses = loss.to(self.device)


                #print(f'Batch Losses: {batch_losses}')
             

                batch_loss = batch_losses.mean()
                
                #print(f'Batch Loss: {batch_loss}')

    
                if mode == 'train':
                    
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.scheduler.step()
                    torch.cuda.synchronize()

                if epoch_count + batch_count != 0:
                    epoch_loss = (epoch_loss * epoch_count + batch_loss.item() * batch_count) / (epoch_count + batch_count)

                    #print(f'Epoch Batch Loss: {batch_loss}')
        
                else:
                     epoch_loss += batch_loss.item()
                     
                     #print(f'Epoch Batch Loss: {batch_loss}')

                epoch_count += batch_count    
        
        # Calculate AUC
        # Concatenate the lists of labels and logits into single tensors
        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        

        return epoch_loss, all_labels, all_logits 
        

    def run(self, epochs=500):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()     
            train_epoch_loss,  train_all_labels, train_all_logits = self.run_epoch(self.train_dataloader, mode='train')
            train_macro_auc, train_micro_auc = calculate_auc(output = train_all_logits, labels= train_all_labels, num_labels=4)
            train_epoch_metrics = [train_macro_auc, train_micro_auc]
            epoch_end_time = datetime.now()
            elapsed = epoch_end_time - epoch_start_time
            elapsed_time=str(elapsed).split('.')[0]

            # Evaluation metrics
            print('###### Saving metrics for training dataset')
            train_metrics = calculate_metrics(output = train_all_logits, labels = train_all_labels, num_labels = 4, epoch = self.epoch, epoch_loss = train_epoch_loss, elapsed= elapsed_time)
            
            if epoch >= 0:
                self.history.append(train_metrics)
                with open(self.checkpoint_dir+'train_metrics.txt', 'a+') as f:
                    f.write(str(train_metrics) + '\n')

            #torch.cuda.empty_cache()
            
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss, val_all_labels, val_all_logits = self.run_epoch(self.valid_dataloader, mode='val')
                val_macro_auc, val_micro_auc = calculate_auc(output = val_all_logits, labels= val_all_labels, num_labels=4)
                val_epoch_metrics = [val_macro_auc, val_micro_auc]

                # Validation evaluation
                print('###### Saving metrics for validation dataset')

                val_metrics = calculate_metrics(output = val_all_logits, labels = val_all_labels, num_labels = 4, epoch = self.epoch, epoch_loss = val_epoch_loss, elapsed= self._elapsed_time())
            
                if epoch >= 0:
                    self.history.append(val_metrics)
                    with open(self.checkpoint_dir+'val_metrics.txt', 'a+') as f:
                        f.write(str(val_metrics) + '\n')



            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                    test_epoch_loss, test_all_labels, test_all_logits = self.run_epoch(self.test_dataloader, mode='test')
                    test_macro_auc, test_micro_auc = calculate_auc(output = test_all_logits, labels= test_all_labels, num_labels=4)
                    test_epoch_metrics = [test_macro_auc, test_micro_auc]
    
                    print(colored('\n Test_MacroAUC (%s) , Test_MicroAUC (%s) ' % (test_macro_auc, test_micro_auc), 'red'))

                    #Evaluation metrics
                    print('###### Saving metrics for testing dataset')
                    test_metrics = calculate_metrics(output = test_all_logits, labels = test_all_labels, num_labels = 4, epoch = self.epoch, epoch_loss = test_epoch_loss, elapsed= self._elapsed_time())
            
                    if epoch >= 0:
                        self.history.append(test_metrics)
                        with open(self.checkpoint_dir+'test_metrics.txt', 'a+') as f:
                            f.write(str(test_metrics) + '\n')
      
            
            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = LOG_FORMAT.format(epoch=epoch,
                                                progress=epoch / epochs,
                                                per_second=per_second,
                                                train_loss=train_epoch_loss,
                                                val_loss=val_epoch_loss,
                                                train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
                                                val_metrics=[round(metric, 4) for metric in val_epoch_metrics],
                                                current_lr=current_lr,
                                                elapsed=self._elapsed_time()
                                                )

                self.logger.info(log_message)
               
        
            representative_val_metric = val_epoch_metrics[1]
            if self.best_val_metric is None or self.best_val_metric < representative_val_metric:
                self.best_val_metric = representative_val_metric
                self.val_metrics_at_best = val_epoch_metrics
                self.val_loss_at_best = val_epoch_loss
                self.train_metrics_at_best = train_epoch_metrics
                self.train_loss_at_best = train_epoch_loss
                self.best_epoch = self.epoch
                self.best_model = self.model
             
                print(colored('best epoch  %d' % self.best_epoch, 'green'))
                print(colored('train loss at best %f' % self.train_loss_at_best, 'green'))
                print(colored('val loss at best %f' % self.val_loss_at_best, 'green'))
       

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics, test_epoch_metrics)
                wandb.log({ "epoch":epoch,
                           "best epoch":self.best_epoch,
                            "train loss":train_epoch_loss,
                            "val loss":val_epoch_loss,
                            "test loss":test_epoch_loss,
                            "label#0 train AUC": train_metrics['AUC'][0],
                            "label#0 train precision": train_metrics['precision'][0],
                            "label#0 train recall": train_metrics['recall'][0],
                            "label#0 train F1": train_metrics['f1'][0],
                            "label#1 train AUC": train_metrics['AUC'][1],
                            "label#1 train precision": train_metrics['precision'][1],
                            "label#1 train recall": train_metrics['recall'][1],
                            "label#1 train F1": train_metrics['f1'][1],
                            "label#2 train AUC": train_metrics['AUC'][2],
                            "label#2 train precision": train_metrics['precision'][2],
                            "label#2 train recall": train_metrics['recall'][2],
                            "label#2 train F1": train_metrics['f1'][2],
                            "label#3 train AUC": train_metrics['AUC'][3],
                            "label#3 train precision": train_metrics['precision'][3],
                            "label#3 train recall": train_metrics['recall'][3],
                            "label#3 train F1": train_metrics['f1'][3],
                            "train macro-AUC":train_metrics['macro_AUC'],
                            "train micro-AUC":train_metrics['micro_AUC'],
                            "train macro-precision":train_metrics['macro_precision'],
                            "train micro-precision":train_metrics['micro_precision'],
                            "train macro-recall":train_metrics['macro_recall'],
                            "train micro-recall":train_metrics['micro_recall'],
                            "train macro-f1":train_metrics['macro_f1'],
                            "train micro-f1":train_metrics['micro_f1'],
                            "label#0 val AUC": val_metrics['AUC'][0],
                            "label#0 val precision": val_metrics['precision'][0],
                            "label#0 val recall": val_metrics['recall'][0],
                            "label#0 val F1": val_metrics['f1'][0],
                            "label#1 val AUC": val_metrics['AUC'][1],
                            "label#1 val precision": val_metrics['precision'][1],
                            "label#1 val recall": val_metrics['recall'][1],
                            "label#1 val F1": val_metrics['f1'][1],
                            "label#2 val AUC": val_metrics['AUC'][2],
                            "label#2 val precision": val_metrics['precision'][2],
                            "label#2 val recall": val_metrics['recall'][2],
                            "label#2 val F1": val_metrics['f1'][2],
                            "label#3 val AUC": val_metrics['AUC'][3],
                            "label#3 val precision": val_metrics['precision'][3],
                            "label#3 val recall": val_metrics['recall'][3],
                            "label#3 val F1": val_metrics['f1'][3],
                            "val macro-AUC":val_metrics['macro_AUC'],
                            "val micro-AUC":val_metrics['micro_AUC'],
                            "val macro-precision":val_metrics['macro_precision'],
                            "val micro-precision":val_metrics['micro_precision'],
                            "val macro-recall":val_metrics['macro_recall'],
                            "val micro-recall":val_metrics['micro_recall'],
                            "val macro-f1":val_metrics['macro_f1'],
                            "val micro-f1":val_metrics['micro_f1'],
                            "label#0 test AUC": test_metrics['AUC'][0],
                            "label#0 test precision": test_metrics['precision'][0],
                            "label#0 test recall": test_metrics['recall'][0],
                            "label#0 test F1": test_metrics['f1'][0],
                            "label#1 test AUC": test_metrics['AUC'][1],
                            "label#1 test precision": test_metrics['precision'][1],
                            "label#1 test recall": test_metrics['recall'][1],
                            "label#1 test F1": test_metrics['f1'][1],
                            "label#2 test AUC": test_metrics['AUC'][2],
                            "label#2 test precision": test_metrics['precision'][2],
                            "label#2 test recall": test_metrics['recall'][2],
                            "label#2 test F1": test_metrics['f1'][2],
                            "label#3 test AUC": test_metrics['AUC'][3],
                            "label#3 test precision": test_metrics['precision'][3],
                            "label#3 test recall": test_metrics['recall'][3],
                            "label#3 test F1": test_metrics['f1'][3],
                            "test macro-AUC":test_metrics['macro_AUC'],
                            "test micro-AUC":test_metrics['micro_AUC'],
                            "test macro-precision":test_metrics['macro_precision'],
                            "test micro-precision":test_metrics['micro_precision'],
                            "test macro-recall":test_metrics['macro_recall'],
                            "test micro-recall":test_metrics['micro_recall'],
                            "test macro-f1":test_metrics['macro_f1'],
                            "test micro-f1":test_metrics['micro_f1'],
                            "current lr": current_lr,
                            "elapsed time": elapsed_time})

        #print(colored('BestValidAuc %f has a TrainAUC %f and TestAuc of %f at epoch %d ' % (self.best_val_metric[1], self.train_metrics_at_best[1], test_macro_auc, self.best_epoch), 'green'))
        
    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics, test_epoch_metrics):
        
        elapsed=self._elapsed_time()
        checkpoint_name = SAVE_FORMAT.format(
            epoch=epoch,
            val_loss= val_epoch_loss,
            val_metrics='-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
        )

        checkpoint_output_path = join(self.checkpoint_dir, checkpoint_name)

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metrics': train_epoch_metrics,
            'val_loss': val_epoch_loss,
            'val_metrics': val_epoch_metrics,
            'test_metrics':test_epoch_metrics,
            'checkpoint': checkpoint_output_path,
            'time cost': elapsed
        }
     
        if epoch >= 0:
            self.history.append(save_state)
            with open(self.checkpoint_dir+'output_history.txt', 'a+') as f:
                f.write(str(save_state) + '\n')

        if hasattr(self.model, 'module'):  # DataParallel
            save_state['state_dict'] = self.model.module.state_dict()
        else:
            save_state['state_dict'] = self.model.state_dict()

        torch.save(save_state, checkpoint_output_path)

        representative_val_metric = val_epoch_metrics[1]
        if self.best_val_metric is None or self.best_val_metric < representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.val_metrics_at_best = val_epoch_metrics
            self.val_loss_at_best = val_epoch_loss
            self.train_metrics_at_best = train_epoch_metrics
            self.train_loss_at_best = train_epoch_loss
            self.best_checkpoint_output_path = checkpoint_output_path
            self.best_epoch = epoch
        
        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_output_path))
            self.logger.info("Current best epoch is {}".format(self.best_epoch))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds
## The inputs will be the training, testing, and validation datasets and the location of the pre-trained model ##
## The config.json file must be in the same folder where this script is ##


def finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", required=True, type=str, help="test dataset for train bert")
    parser.add_argument("-v", "--valid_dataset", required=True, type=str, help="valid dataset for train bert")
    
    parser.add_argument("-p", "--pretrainedmodel_path", required=True, type=str, default=str, help="pretrained model path")
    parser.add_argument("-config", "--config_path", required=True, type=str, default=str, help="config json file")
    parser.add_argument("-vocab", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
   
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--log_output', type=str, default=None)
    
    parser.add_argument("-hs", "--hidden_size", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--num_hidden_layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--num_attention_heads", type=int, default=6, help="number of attention heads")
    parser.add_argument("-s", "--max_len", type=int, default=512, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")
    
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false") 
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument('--clip_grads', action='store_true')
    
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    print('Class & Loss Weights')
    class_weights = torch.tensor([0.0004,0.0003,0.0003,0.001], dtype=torch.float32)
    #loss_weights = torch.tensor([3.38,2.29,2.35,9.28], dtype=torch.float32)
    loss_weights = torch.tensor([12.23,8.27,8.37,36.45], dtype=torch.float32)
       
       
    print('Loading config file from the pretrained model')
    
    # Load the configuration from a pretrained model
    config = BertConfig.from_json_file(args.config_path)
    config_dict = config.__dict__
    

    run_name = None
    run_name = run_name if run_name is not None else make_run_name(RUN_NAME_FORMAT, phase='pretrain', config=config_dict)
    logger = make_logger(run_name, args.log_output)
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Loading vocab file')
    vocab = pkl.load(open(args.vocab_path, 'rb'), encoding='bytes')
    vocab_size = len(vocab)
    print("Vocab Size: ", vocab_size)

    logger.info('Loading training, validation, and testing dataset')
    train_f = pkl.load(open(args.train_dataset, 'rb'), encoding='bytes')
    valid_f=pkl.load( open(args.valid_dataset, 'rb'), encoding='bytes')
    test_f=pkl.load( open(args.test_dataset, 'rb'), encoding='bytes')
    
    logger.info('Creating BERT Features')
    train_features = convert_EHRexamples_to_features(train_f, args.max_len) 
    valid_features = convert_EHRexamples_to_features(valid_f, args.max_len)
    test_features = convert_EHRexamples_to_features(test_f, args.max_len)
    
    train = BERTdataEHR(train_features)
    valid = BERTdataEHR(valid_features)
    test = BERTdataEHR(test_features)
      
    logger.info('Creating Dataloader')
    train_dataloader = BERTdataEHRloader(train, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = BERTdataEHRloader(valid, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = BERTdataEHRloader(test, batch_size=args.batch_size, num_workers=args.num_workers)
 
    logger.info('Building BERT model for pretraining')
    pretrained_model = BertForPreTraining(config)
    model_load = torch.load(args.pretrainedmodel_path)
    state_dict = model_load['state_dict']
    pretrained_model.load_state_dict(state_dict)
    #pretrained_model = BertForSequenceClassification.from_pretrained(args.pretrainedmodel_path, num_labels=4)
    model = EHRBertMultiOutputClassifier(config, pretrained_model,  num_labels=4)

    logger.info(model)
    logger.info('{parameters_count} parameters'.format(
        parameters_count=sum([p.nelement() for p in model.parameters()])))

    loss_model = BertFineTuningCriterion(num_labels=4, loss_weights=loss_weights)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)



    checkpoint_dir = make_checkpoint_dir(args.checkpoint_dir, run_name, config)

    logger.info('Start training...')
    trainer = BertTrainer(
        model = model,
        loss_model=loss_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler = scheduler,
        clip_grads=args.clip_grads,
        logger=logger,
        checkpoint_dir=args.checkpoint_dir,
        print_every=args.print_every,
        save_every=args.save_every,
        with_cuda = True
    )

    trainer.run(epochs=args.epochs)

if __name__ == "__main__":
    finetune()
    wandb.finish()









    
