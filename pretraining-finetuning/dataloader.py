import pandas as pd
import random
import numpy as np
from datetime import datetime
import pickle as pkl
import os, sys
import argparse
import torch 
from torch.utils.data import Dataset, DataLoader

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               masked_input_ids,
               label_id,
               patient_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.masked_input_ids = masked_input_ids
    self.label_id = label_id
    self.patient_id = patient_id
    self.is_real_example = is_real_example

class BERTdataEHR(Dataset):
    def __init__(self, Features):
        self.data = Features

    def __getitem__(self, idx, seeDescription=False):
        sample = self.data[idx]
        # Create a dictionary to store the features
        feature_dict = {
            "input_ids": sample.input_ids,
            "input_mask": sample.input_mask,
            "segment_ids": sample.segment_ids,
            "masked_input_ids": sample.masked_input_ids,
            "label_id": sample.label_id,
            "patient_id": sample.patient_id,
            "is_real_example": sample.is_real_example
        }
        #return sample
        return feature_dict
    def __len__(self):
        return len(self.data)

def my_collate(batch):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_masked_input_ids = []
    all_label_ids = []
    all_patient_ids = []
    """
    for feature in batch:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_masked_input_ids.append(feature.masked_input_ids)
        all_label_ids.append(feature.label_id)
        all_patient_ids.append(feature.patient_id)

    return [all_input_ids, all_input_mask, all_segment_ids, all_masked_input_ids, all_label_ids, all_patient_ids]
    """
    for feature in batch:
        all_input_ids.append(feature["input_ids"])
        all_input_mask.append(feature["input_mask"])
        all_segment_ids.append(feature["segment_ids"])
        all_masked_input_ids.append(feature["masked_input_ids"])
        all_label_ids.append(feature["label_id"])
        all_patient_ids.append(feature["patient_id"])

    return {
        "input_ids": torch.tensor(all_input_ids),
        "input_mask": torch.tensor(all_input_mask),
        "segment_ids": torch.tensor(all_segment_ids),
        "masked_input_ids": torch.tensor(all_masked_input_ids),
        "label_id": torch.tensor(all_label_ids),
        "patient_id": torch.tensor(all_patient_ids)
    }
class BERTdataEHRloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                            timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn



