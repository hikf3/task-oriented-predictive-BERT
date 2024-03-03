#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:57:09 2023

@author: hikf3

Add the output labels in the list at the second position
"""
import pandas as pd
import pickle as pkl
#import gzip

train_f=pkl.load( open('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/seq_features_train', 'rb'), encoding='bytes')
# Access the desired row (e.g., row 0)
#row = train_f[100]

# Print the row
#print(row)


valid_f=pkl.load( open('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/seq_features_valid', 'rb'), encoding='bytes')
test_f=pkl.load( open('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/seq_features_test', 'rb'), encoding='bytes')
print('Printing length of Training dataset', len(train_f))
print('Printing length of Testing dataset', len(test_f))
print('Printing length of Validation dataset', len(valid_f))


def add_labelList(labelPath, featSeq, dataType, year ):
    # Load labels.csv.gz as a Pandas dataframe
    df_labels = pd.read_csv(labelPath, compression='gzip')
    # Extract columns 2 to 5 (index 1 to 4) into a list
    label_list = df_labels.iloc[:, 0:5].values.tolist()
    
    # Create a dictionary of IDs and label lists
    label_dict = {str(row[0]): row[1:] for row in label_list}
    
    
    # Insert the label lists into the second position of each row in train_f
    for i, row in enumerate(featSeq):
        id = str(row[0])
        if id in label_dict:
            featSeq[i].insert(1, label_dict[id])
            #featSeq[i].insert(1, [])
            
    # Print the modified train_f
    #for row in featSeq:
        #print(row)
     # Save the updated pickled list
    print(len(featSeq[0]))
    with open('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/'+'FINETUNING_LIST_'+ year + '_'+ dataType +'.pkl', 'wb') as f:
        pkl.dump(featSeq, f)  

# Year 1
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR1_TRAIN.csv.gz', train_f, 'TRAIN', 'YR1')
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR1_TEST.csv.gz', test_f, 'TEST', 'YR1')
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR1_VALID.csv.gz', valid_f, 'VALID', 'YR1')

# Year 2
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR2_TRAIN.csv.gz', train_f, 'TRAIN', 'YR2')
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR2_TEST.csv.gz', test_f, 'TEST', 'YR2')
#add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR2_VALID.csv.gz', valid_f, 'VALID', 'YR2')


# Year 3
add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR3_TRAIN.csv.gz', train_f, 'TRAIN', 'YR3')
add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR3_TEST.csv.gz', test_f, 'TEST', 'YR3')
add_labelList('/Users/hikf3/Documents/GitHub/Doctoral-Research/BEHRT/DATA/TRAIN_TEST_VALID/DX_AGE/LABELS_YR3_VALID.csv.gz', valid_f, 'VALID', 'YR3')






    
