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
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
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
               masked_input_ids,
               nsp_label_id,
               label_ids,
               patient_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.timetonext_ids = timetonext_ids
    self.enc_age_ids = enc_age_ids
    self.segment_ids = segment_ids
    self.visit_segment_ids = visit_segment_ids
    self.masked_input_ids = masked_input_ids
    self.nsp_label_id = nsp_label_id
    self.label_ids = label_ids
    self.patient_id = patient_id
    self.is_real_example = is_real_example

    

    
def convert_EHRexamples_to_features(examples,max_seq_length, vocab):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_singleEHR_example(ex_index, example, max_seq_length, vocab)
        features.append(feature)
    return features

### This is the EHR version

def convert_singleEHR_example(ex_index, example, max_seq_length, vocab):
    
    input_ids = example[4]
    segment_ids = example[7]
    patient_id = example[0]
    timetonext_ids = example[3]
    enc_age_ids = example[5]
    visit_segment_ids = example[6]
    label_ids = example[1]

    #print('before insert', patient_id, len(input_ids), len(segment_ids),  len(timetonext_ids), len(enc_age_ids), len(visit_segment_ids))
    

    # insert [CLS] and adjust the other lists
    input_ids.insert(0,2)
    segment_ids.insert(0,segment_ids[0])
    timetonext_ids.insert(0,timetonext_ids[0])
    enc_age_ids.insert(0,enc_age_ids[0])
    visit_segment_ids.insert(0,visit_segment_ids[0])

    #print('after insert',patient_id, len(input_ids), len(segment_ids),  len(timetonext_ids), len(enc_age_ids), len(visit_segment_ids))
    

    # Next sentence label
    if max(example[2]) > 7:
        nsp_label_id = 1  # Time between 2 visits
    else:
        nsp_label_id = 0
    
    # Masking the input_ids
    masked_input_ids = create_masked_input_ids(input_ids, max_seq_length, vocab)
 
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
        masked_input_ids=masked_input_ids,
        nsp_label_id = nsp_label_id,
        label_ids=label_ids,
        patient_id=patient_id,
        is_real_example=True
    )
    
    return feature

def create_masked_input_ids(input_ids, max_seq_length, vocab):
    masked_input_ids = input_ids.copy()

    # Calculate the number of tokens to mask
    num_tokens_to_mask = int(max_seq_length * 0.15)

    # Determine the number of tokens to change to 0, random tokens, and 1
    num_zeros = int(num_tokens_to_mask * 0.8)
    num_random = int(num_tokens_to_mask * 0.1)
    #num_ones = num_tokens_to_mask - num_zeros - num_random
    
    # Shuffle the input_ids
    #random.shuffle(masked_input_ids)
    # Generate random positions to mask
    positions_to_mask = random.sample(range(max_seq_length), num_tokens_to_mask)
    
    
    # Zero-pad or truncate the masked_input_ids to match max_seq_length
    while len(masked_input_ids) < max_seq_length:
        masked_input_ids.append(0)
    while len(masked_input_ids) > max_seq_length:
        masked_input_ids.pop()

    # Mask the tokens
    for i in range(num_zeros):
        masked_input_ids[positions_to_mask[i]] = -1
    vocab_values = list(vocab.values())
    for i in range(num_zeros, num_zeros + num_random):
        # Generate a random token (starting from 5)
        masked_input_ids[positions_to_mask[i]] = random.randint(4, max(vocab_values) - 1)

    #for i in range(num_zeros + num_random, num_zeros + num_random + num_ones):
        #masked_input_ids[positions_to_mask[i]] = 1

    return masked_input_ids

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
            "masked_input_ids": sample.masked_input_ids,
            "nsp_label_id": sample.nsp_label_id,
            "label_ids": sample.label_ids,
            "patient_id": sample.patient_id,
            "is_real_example": sample.is_real_example
        }
        
        return feature_dict, torch.tensor(sample.label_ids, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

def my_collate(batch):
    all_input_ids = []
    all_input_mask = []
    all_timetonext_ids =[]
    all_enc_age_ids =[]
    all_visit_segment_ids =[]
    all_segment_ids = []
    all_masked_input_ids = []
    all_nsp_label_ids =[]
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
        all_masked_input_ids.append(feature["masked_input_ids"])
        all_nsp_label_ids.append(feature["nsp_label_id"])
        all_label_ids.append(feature["label_ids"]) # Convert 0-dimensional tensor to a scalar
        all_patient_ids.append(feature["patient_id"]) # Convert 0-dimensional tensor to a scalar
        all_labels.append(labels)

    return {
        "input_ids": torch.tensor(all_input_ids),
        "input_mask": torch.tensor(all_input_mask),
        "timetonext_ids": torch.tensor(all_timetonext_ids),
        "enc_age_ids": torch.tensor(all_enc_age_ids),
        "visit_segment_ids": torch.tensor(all_visit_segment_ids),
        "segment_ids": torch.tensor(all_segment_ids),
        "masked_input_ids": torch.tensor(all_masked_input_ids),
        "nsp_label_id": torch.tensor(all_nsp_label_ids),
        "label_ids": torch.tensor(all_label_ids),
        "patient_id": torch.tensor(all_patient_ids),
        "labels": torch.stack(all_labels)
    }
class BERTdataEHRloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                            timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn
def finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--test_dataset", required=True, type=str, help="test dataset for train bert")
    
    parser.add_argument("-p", "--pretrainedmodel_path", required=True, type=str, default=str, help="pretrained model path")
    parser.add_argument("-config", "--config_path", required=True, type=str, default=str, help="config json file")
    parser.add_argument("-vocab", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
   
    parser.add_argument("-d","--checkpoint_dir", type=str, default=None)
    parser.add_argument('--log_output', type=str, default=None)
    
    parser.add_argument("-hs", "--hidden_size", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--num_hidden_layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--num_attention_heads", type=int, default=6, help="number of attention heads")
    parser.add_argument("-s", "--max_len", type=int, default=512, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")
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
    dir = args.checkpoint_dir
    print('directory',dir)
    print('Loading config file from the pretrained model')
    
    # Load the configuration from a pretrained model
    config = BertConfig.from_json_file(args.config_path)
    config_dict = config.__dict__
    vocab = pkl.load(open(args.vocab_path, 'rb'), encoding='bytes')

    
    print('Loading training, validation, and testing dataset')
    test_f=pkl.load( open(args.test_dataset, 'rb'), encoding='bytes')
    
    
    print('Creating BERT Features')
    test_features = convert_EHRexamples_to_features(test_f, args.max_len, vocab)
    test = BERTdataEHR(test_features)
      
    print('Creating Dataloader')
    test_dataloader = BERTdataEHRloader(test, batch_size=args.batch_size, num_workers=args.num_workers)
 
    print('Loading fine-tuned BERT model for computing attributions')
    pretrained_model = BertForPreTraining(config, num_labels=4)
    model_load = torch.load(args.pretrainedmodel_path)
    state_dict = model_load['state_dict']
    pretrained_model.load_state_dict(state_dict)
    model = pretrained_model
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    model.zero_grad()
    
    # Create an iterator from the dataloader
    iterator = iter(test_dataloader)
    batch = next(iterator)
    input_ids=batch['input_ids'].to(device)
    #print('###INPUT_IDS',input_ids.shape)
    #reference inputs
    # Create a new tensor with the same shape filled with zeros
    original_shape= input_ids.shape
    zero_tensor = torch.zeros(original_shape, dtype=torch.long)
    ref_input_ids = zero_tensor.to(device)
    print('reference ids', zero_tensor.shape)

    
    def forward_func(input_ids,  visit_ids=None, timetonext_ids=None, attention_mask=None):
        prediction_scores, seq_relationship_score, logits = model(input_ids, visit_ids, timetonext_ids, attention_mask)
        return(logits[:,1])
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    
    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions
    
    # Initialize a list to store all results
    all_results = []
    batch_iter = tqdm(
            test_dataloader,
            desc="Iteration",
            disable=False,
            total=len(test_dataloader))

        
    for batch_count, batch in enumerate(batch_iter):
                #0. batch_data will be sent into the device(GPU or cpu)
                batch = {key: value.to(device) for key, value in batch.items()}
                input_ids = batch['input_ids']
                visit_ids = batch['visit_segment_ids']
                timetonext_ids = batch['timetonext_ids']
                attention_mask = batch['input_mask']

                attributions, delta_start = lig.attribute(inputs = input_ids,
                                  baselines=ref_input_ids,
                                  additional_forward_args=(visit_ids, timetonext_ids, attention_mask),
                                  return_convergence_delta=True)
                attributions_sum = summarize_attributions(attributions)
                patient_id = batch['patient_id']
                #print('ATTRIBUTIONS', attributions_sum.shape)
            
                    
                # Convert necessary information to CPU and numpy, and reshape as needed
                patient_id_np = patient_id.cpu().numpy()
                attributions_list = attributions_sum.cpu().tolist()
        
                # Combine all results for this batch into a single array
                for i, patient_id  in enumerate(patient_id_np):
                        row = [patient_id] + attributions_list
                        all_results.append(row)

    # Create a DataFrame from the results
    # Assuming the shape[0] of attributions_np gives the number of attribution values per patient_id
    num_attributions = len(attributions_list) # Corrected to access the second dimension for attributions per patient_id

    columns = ['patient_id'] + \
              [f'attributions_np_{i}' for i in range(num_attributions)] 
              
    results_df = pd.DataFrame(all_results, columns=columns)

    # Save the DataFrame to a CSV file
    csv_file_path = f'{dir}_MACE_attributions.csv'
    results_df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
       

if __name__ == "__main__":
    finetune()