#import pandas as pd
import random
#import numpy as np
from datetime import datetime
import pickle as pkl
import os, sys
import argparse
import torch 
from torch.utils.data import Dataset

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               timetonext_ids,
               enc_age_ids, 
               segment_ids,
               visit_segment_ids,
               masked_input_ids,
               label_id,
               patient_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.timetonext_ids = timetonext_ids
    self.enc_age_ids = enc_age_ids
    self.segment_ids = segment_ids
    self.visit_segment_ids = visit_segment_ids
    self.masked_input_ids = masked_input_ids
    self.label_id = label_id
    self.patient_id = patient_id
    self.is_real_example = is_real_example

def convert_EHRexamples_to_features(examples, max_seq_length, vocab):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_singleEHR_example(ex_index, example, max_seq_length, vocab)
        features.append(feature)
    return features

def convert_singleEHR_example(ex_index, example, max_seq_length, vocab):
    input_ids = example[3]
    segment_ids = example[6]
    patient_id = example[0]
    timetonext_ids = example[2]
    enc_age_ids = example[4]
    visit_segment_ids = example[5]

    # insert [CLS] and adjust the other lists
    input_ids.insert(0,2)
    segment_ids.insert(0,segment_ids[0])
    timetonext_ids.insert(0,timetonext_ids[0])
    enc_age_ids.insert(0,enc_age_ids[0])
    visit_segment_ids.insert(0,visit_segment_ids[0])



    # Next sentence label
    if max(example[1]) > 7:
        label_id = 1  # Time between 2 visits
    else:
        label_id = 0

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
    
    '''# The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Masking the input_ids
    masked_input_ids = create_masked_input_ids(input_ids, max_seq_length, vocab)

    # Left Truncate longer sequence
    while len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        input_mask = input_mask[-max_seq_length:]
        timetonext_ids = timetonext_ids[-max_seq_length:]
        enc_age_ids = enc_age_ids[-max_seq_length:]
        segment_ids = segment_ids[-max_seq_length:]


    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        timetonext_ids.append(0)
        enc_age_ids.append(0)
        segment_ids.append(0)

   '''

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(timetonext_ids) == max_seq_length
    assert len(enc_age_ids) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(visit_segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        timetonext_ids=timetonext_ids,
        enc_age_ids=enc_age_ids,
        visit_segment_ids = visit_segment_ids,
        segment_ids=segment_ids,
        masked_input_ids=masked_input_ids,
        label_id=label_id,
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

class BERTdataEHR(Dataset):
    def __init__(self, Features):
        self.data = Features

    def __getitem__(self, idx, seeDescription=False):
        sample = self.data[idx]
        # Create a dictionary to store the features
        feature_dict = {
            "input_ids": torch.tensor(sample.input_ids),
            "input_mask": torch.tensor(sample.input_mask),
            "timetonext_ids" : torch.tensor(sample.timetonext_ids),
            "enc_age_ids": torch.tensor(sample.enc_age_ids),
            "visit_segment_ids": torch.tensor(sample.visit_segment_ids),
            "segment_ids": torch.tensor(sample.segment_ids),
            "masked_input_ids": torch.tensor(sample.masked_input_ids),
            "label_id": torch.tensor(sample.label_id),
            "patient_id": torch.tensor(sample.patient_id),
            "is_real_example": torch.tensor(sample.is_real_example)
        }
        return feature_dict
    def __len__(self):
        return len(self.data)
    

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-v", "--vocab_file", required=True, type=str, help="vocabulary file for DX")
    parser.add_argument("-s", "--max_seq_length", type=int, default=512, help="maximum sequence length")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="path to save output files")
    args = parser.parse_args()

    train_f = pkl.load(open(args.train_dataset, 'rb'), encoding='bytes')
    vocab_file = pkl.load(open(args.vocab_file, 'rb'), encoding='bytes')
    max_seq_length = args.max_seq_length

    print('Convert EHR seq to Examples')
    train_features = convert_EHRexamples_to_features(train_f, max_seq_length, vocab_file)

    num_zeros = sum(feature.label_id == 0 for feature in train_features)
    num_ones = sum(feature.label_id == 1 for feature in train_features)

    print("Total number of label_id=0: ", num_zeros)
    print("Total number of label_id=1: ", num_ones)

    print("Creating BERT PyTorch Dataset")
    train = BERTdataEHR(train_features)
    #print('#Printing some examples',train_features[0].label_id)
    # Open the file in binary mode and save the train_features using pickle.dump()
    outFile= args.output_path + 'bert_dataset_train.pkl'
    with open(outFile, 'wb') as file:
        pkl.dump(train, file)

    print("Train features saved as a pickle file:", outFile)
   
    


if __name__ == "__main__":
    build()
    