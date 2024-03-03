#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 2023

@author: hikf3

convert train test valid data into sequences
"""

import sys
import pickle
import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from math import ceil

### Creating the sequence for each patid
def seq_fn(pts_sls, outFile, subset):
    print("Creating sequence for each PATID")
    dataSize = len(pts_sls)
    np.random.seed(0)
    ind = np.random.permutation(dataSize)
    indices = ind
    subset_ptencs_s = [pts_sls[i] for i in indices]
    bertencsfile = outFile  + subset
    pickle.dump(subset_ptencs_s, open(bertencsfile, 'a+b'), -1)

### Main Function
if __name__ == '__main__':
    diagFile = sys.argv[1]
    print('DX_AGE_data', sys.argv[1])
    typeFile = sys.argv[2]
    print('Mapping File', sys.argv[2])
    outFile = sys.argv[3]
    print('Name given', sys.argv[3])
    subset = sys.argv[4]
    print('Data Subset', sys.argv[4])

    # Data Loading
    print("Data file loading")
    data_diag = pd.read_csv(diagFile)
    data_diag.columns = ['patient_sk', 'admit_dt_tm', 'discharge_dt_tm', 'diagnosis', 'poa', 'third_party_ind', 'diagnosis_priority', 'dob', 'age_at_enc', 'sex']


    # Loading dictionary that maps different Phe codes to integer values.
    # If you have a pre-existing vocab file, you can use the path for such a file or
    # you can use NA to create a new one.
  
    with open(typeFile, 'rb') as t2:
            types = pickle.load(t2)

    n_data = data_diag
    count = 0
    pts_sls = []

    # Looping through the data grouped by patient identifier
    for Pt, group in n_data.groupby('patient_sk'):
        pt_encs = []
        time_tonext = []
        pt_los = []
        full_seq = []
        v_seg = []
        age_seq = []
        pt_discdt = []
        pt_addt = []
        v = 0
        len_dx =[]

        # For each patient, the code creates a list of encounters, grouped by discharge date.
        for Time, subgroup in group.sort_values(['admit_dt_tm', 'poa', 'third_party_ind', 'diagnosis_priority'], ascending=True).groupby('admit_dt_tm', sort=False):
            v = v + 1
            #print('v', v)

            

            subgroup['diagnosis'] = subgroup['diagnosis'].apply(lambda x: 'phe' + str(x)) 
            diag_l = np.array(subgroup['diagnosis'].drop_duplicates()).tolist()

            if len(diag_l) > 0:
                diag_lm = []

                # If the code exists in the types dictionary, convert diagnosis codes to integer codes
                # and add them to the full_seq. If the code does not exist in types dict,
                # add the code as a new key to the dict with the value being the maximum value in the dict plus one.
                for code in diag_l:
                    if code in types:
                        diag_lm.append(types[code])
                    else:
                        types[code] = max(types.values()) + 1
                        diag_lm.append(types[code])
                    
                    v_seg.append(v)
                    #print('visit', len(v_seg)) 
                #print('length of diag_lm',len(diag_lm))
                len_dx.append(len(diag_lm))
                #print('lenght of len_dx', len(len_dx))
                age_at_encounter = np.repeat(ceil(subgroup['age_at_enc'].values[0]*12), len(diag_lm)).tolist()
                age_seq.extend(age_at_encounter)
                #print('age_seq', len(age_seq)) 

                full_seq.extend(diag_lm)
        
    
            pt_addt.append(dt.strptime(Time, '%Y-%m-%d'))
            pt_discdt.append(dt.strptime(min(np.array(subgroup['discharge_dt_tm'].drop_duplicates()).tolist()), '%Y-%m-%d'))

        if len(pt_addt) > 0:
            for ei, eid in enumerate(pt_addt):
                if ei == len(pt_addt) - 1:
                    enc_td = 0 
                else:
                    enc_td = ceil((pt_addt[ei + 1] - pt_addt[ei]).days /30.44) # Convert days to months, rounding to nearest integer
                enc_los = (pt_discdt[ei] - pt_addt[ei]).days

                time_tonext.append(enc_td)
                #print('time to next',len(time_tonext))
                pt_los.append(enc_los)

        # Initialize an empty list to store the repeated values of timetonext
        repeated_values_timetonext = []
        segment =[]
        
        len_rep = len(len_dx)
        num_rep = (len_rep + 1) //2
        seg = np.tile([1,2], num_rep)[:len_rep]

        # Loop through both lists simultaneously
        for time, repeat_count in zip(time_tonext, len_dx):
            repeated_values_timetonext.extend([time] * repeat_count)

        for seg, repeat_count in zip(seg, len_dx):
            segment.extend([seg] * repeat_count)


        pts_sls.append([Pt, pt_los, repeated_values_timetonext, full_seq, age_seq, v_seg, segment])
     
        

        count = count + 1

        if count % 1000 == 0:
            print('processed %d pts' % count)

        if count % 100000 == 0:
            print('dumping %d pts' % count)
            seq_fn(pts_sls, outFile)
            pts_sls = []

    seq_fn(pts_sls, outFile, subset)
   
    

    # This line is used to store the information in the "types" variable to a binary file
    #pickle.dump(types, open(outFile + '.types', 'wb'), -1)
