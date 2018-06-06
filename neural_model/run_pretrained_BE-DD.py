# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:01:46 2018
@author: aymm
Based on tutorial in https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
Takes in 2 files as arguments. The first file is the training data, and the second file is the test data. 
The program reads in the files, builds a neural model based on Tensorflow, dynamically trains on the training data
in batches for a number of epochs (as determined by the EPOCHS variable), and test the trained model on the test data.
Data is expected to be in the following format and aligned by character:
Gold standard line 1
OCR output line 1
Gold standard line 2
OCR output line 2
...
Gold standard line x
OCR output line x
Note that the program will crash if the data is not aligned by character!
"""

import numpy as np
import tensorflow as tf
import time
import json
import os
import sys

"""
Load and processes data from the given data files. Separates the lines into
OCR output (X/input data) and gold standard (Y/label data). Prints out the 
lengths of the datasets make sure sets are properly aligned. Returns the 
X and Y sets.
"""
        
def get_data(file_name):
    with open(file_name,'r') as f:
        raw_x = []
        
        for line in f.readlines():
            line = line.strip() + '\n'
            line = line.decode('utf-8', 'replace')
            if line:
                raw_x.append(list(line))
		
        #print("X Data Length:", len(raw_x))

    return raw_x
        

    
"""
Resets tensorflow for new trained model if one has already been trained.
"""
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()    
        


"""
Takes in the trained model and returns the predicted output of the test data.
Note that preds is a matrix that is of the shape (test_data, vocab_size), where each
array is a distribution for each character in the test_data and each index in the array
is the probability that each character in the training vocabulary is the output. 
"""
def eval_network(ckpt, idx_to_vocab):
    with tf.Session() as sess:
        ckpt_file = tf.train.latest_checkpoint(ckpt)
        saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
        saver.restore(sess, ckpt_file)
        g = tf.get_default_graph()
        init_op = tf.global_variables_initializer()
        batch_size = g.get_tensor_by_name("encoder/batch_size:0")

        x = g.get_tensor_by_name("encoder/input_placeholder:0")
        y = g.get_tensor_by_name("encoder/labels_placeholder:0")
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
        data_iter = g.get_operation_by_name("encoder/Iterator")
        make_iter = g.get_operation_by_name("encoder/MakeIterator")
        sess.run(make_iter, feed_dict={x: test_X_data, 
                                       y: test_X_data, 
                                       batch_size: len(test_X_data)})
        preds = g.get_tensor_by_name("decoder/preds:0")
        output = sess.run(preds)#, feed_dict={x: test_X_data, y: test_Y_data})
        idx = np.where(output == 0)
        new_out = np.delete(output, idx)
        mapping = lambda t: idx_to_vocab[str(t)]
        char_func = np.vectorize(mapping)
        chars = char_func(new_out)
        return "".join(chars)    

            
    
# MAIN
test_file = sys.argv[1]
CHECKPOINT = "neural_model/pretrained/bidirectENC_dynamDEC/"
vocab_idx_file = CHECKPOINT + "5-25-2018-vocab-dictionaries"

idx_vocab = dict()
vocab_idx = dict()
tr_vocab_size = 0

with open(vocab_idx_file) as VI_json_file:  
    vocab_tuple = json.load(VI_json_file)
    idx_vocab = vocab_tuple[0]
    vocab_idx = vocab_tuple[1]
    tr_vocab_size = vocab_tuple[2]


test_raw_x = get_data(test_file)

MAX_LEN = 133
EDIT_SPACE= 5

test_X_data = [[vocab_idx[c] for c in arr] for arr in test_raw_x]

test_Y_data = np.array([np.pad(line, (0, MAX_LEN-len(line)+EDIT_SPACE), 'constant', constant_values=0) for line in test_X_data])
test_X_data = np.array([np.pad(line, (0, MAX_LEN-len(line)), 'constant', constant_values=0) for line in test_X_data])

EPOCHS = 20
BATCH_SIZE = 100

chars = eval_network(CHECKPOINT, idx_vocab)
outfile = test_file.split(".")[0] + ".pred"
print(chars.encode('utf-8').strip())
