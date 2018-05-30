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
X and Y sets as matrices and all of the data as a string.
"""
        
def get_data(file_name):
    max_len = 0
    all_data = ""
    raw_x = []
    raw_y = []
    
    with open(file_name,'r') as f:
       	count = 1
       	for line in f.readlines():
            line = line.strip() + '\n'
            line = line.decode('utf-8', 'replace')
            if count % 3 == 1:
                raw_y.append(list(line))
                last_y = len(line)
            elif count % 3 == 2:
                raw_x.append(list(line))
                last_x = len(line)
                
            count += 1
            if len(line) > max_len:
                max_len = len(line)
                        
            all_data += line

    print(file_name, max_len)
    return raw_x, raw_y, all_data, max_len
        
        
"""
Takes in the X and Y datasets and represents each character in both sets as an integer.
Note that these are the only characters that the model will be able to predict.
Returns dictionaries mapping the characters to the representative integers and the integers
to the characters, as well as the number of unique characters.
"""
def make_train_dict(tr_data):
    vocab = set(tr_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict()
    idx_to_vocab[0] = "<PAD>"
    count = 1
    for char in vocab:
        idx_to_vocab[count] = char
        count += 1

    vocab_size = len(idx_to_vocab)
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    
    return idx_to_vocab, vocab_to_idx, vocab_size
    
    
"""
Resets tensorflow for new trained model if one has already been trained.
"""
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()    
    
    
"""
Builds and returns a bidirectional encoder-decoder graph with capabilities for dynamically inputing data
"""
def build_multilayer_lstm_graph_with_dynamic_rnn(
    num_classes, 
    state_size = 100,
    embedding_size=256,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()
    with tf.variable_scope('encoder'):
        batch_size = tf.placeholder(tf.int64, name='batch_size')

        x = tf.placeholder(tf.int32, shape=(None, None), name='input_placeholder')
        y = tf.placeholder(tf.int32, shape=(None, None), name='labels_placeholder')

    # The following allows the model to dynamically accept different sizes of x and also 
    # produces batches of the input data. 
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
    
        data_iter = dataset.make_initializable_iterator()
        features, labels = data_iter.get_next()
    
        embeddings = tf.get_variable('embedding_matrix', [num_classes, embedding_size])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, features)
        dec_inputs = tf.nn.embedding_lookup(embeddings, labels)
    
	# ENCODING -----

        fw_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        """
        TODO: Test how dropout affects accuracy
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=global_dropout)
        """
        
        dynam_batch_size = tf.shape(features)[0]
        fw_init_state = fw_cell.zero_state(dynam_batch_size, tf.float32)
        bw_init_state = bw_cell.zero_state(dynam_batch_size, tf.float32)
        (fw_output, bw_output), (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, 
            bw_cell, 
            rnn_inputs, 
            initial_state_fw=fw_init_state, 
            initial_state_bw=bw_init_state)


	# DECODING -----
	# From tutorial https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb

        enc_final_state_c = tf.concat((fw_final_state.c, bw_final_state.c), 1)
        enc_final_state_h = tf.concat((fw_final_state.h, bw_final_state.h), 1)
        enc_final_state = tf.contrib.rnn.LSTMStateTuple(c=enc_final_state_c, 
                                     h=enc_final_state_h)	
    with tf.variable_scope('decoder'):
        W = tf.get_variable('W', [2*state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    

        decoder_cell = tf.nn.rnn_cell.LSTMCell(2*state_size, state_is_tuple=True)
        decoder_init_state = decoder_cell.zero_state(dynam_batch_size, tf.float32)
    
        rnn_outputs, final_state = tf.nn.dynamic_rnn(decoder_cell, dec_inputs, initial_state=enc_final_state)

    #reshape rnn_outputs and y so we can get the logits in a single matmul
        rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*state_size])
        y_reshaped = tf.reshape(labels, [-1])

        logits = tf.matmul(rnn_outputs, W) + b

        predictions = tf.argmax(logits, axis=1, name='preds')

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    # Different optimizer recommended by https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)
    
    # The following are accessible once the graph has been built:
    return dict(
        x = x,
        y = y,
        fw_init = fw_init_state, 
        bw_init = bw_init_state,
        batch_size = batch_size,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        logits = logits,
        data_iter = data_iter,
        saver = tf.train.Saver()
    )
    

        
        

"""
Takes in the built model and trains over the training data for the given number of EPOCHS in the given
BATCH_SIZE. Saves the session for later use after training. Prints the training losses for each epoch if verbose=True.
"""
#TODO: Allow for save filename to be given in command line
def train_network(g, save, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        sess.run(g['data_iter'].initializer, 
                 feed_dict={g['x']: tr_X_data, 
                            g['y']: tr_Y_data, 
                            g['batch_size']: NUM_IN_BATCH})
                                          
        training_losses = []
        for i in range(EPOCHS):    
            steps = 0
            tot_loss = 0
            training_state = None
            for _ in range(NUM_IN_BATCH):
                loss_value, training_state, _ = sess.run([g['total_loss'], 
                                                          g['final_state'], 
                                                          g['train_step']])
                tot_loss += loss_value
                steps += 1

            if verbose:
                print("Average training loss for Epoch", i, ":", tot_loss/steps)
            training_losses.append(tot_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses
    

"""
Takes in the trained model and returns the predicted output of the test data.
Note that preds is a matrix that is of the shape (test_data, vocab_size), where each
array is a distribution for each character in the test_data and each index in the array
is the probability that each character in the training vocabulary is the output. 
"""
def eval_network(g, checkpoint, idx_to_vocab):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)
        
        sess.run(g['data_iter'].initializer, 
                 feed_dict={g['x']: test_X_data, 
                            g['y']: test_Y_data, 
                            g['batch_size']: len(test_X_data)})
        total_loss, preds = sess.run([g['total_loss'], g['preds']])
        print(preds)
        print(preds.shape)
        print("Total test loss = ", total_loss)
        pad_idx = np.where(preds==0)
        nopad_preds = np.delete(preds, pad_idx)
        char_func = np.vectorize(lambda t: idx_to_vocab[t])
        chars = char_func(preds)
        return "".join(chars)
    

"""
Compares the predicted string and the gold standard string. Returns the 
error, the total number of characters, and the accuracy.
"""
def char_err_rate(pred, gold):
    total = 0.
    error = 0.
    for i in range(len(pred)):
        total+= 1
        if i < len(gold):
            if not pred[i] == gold[i]:
                error+= 1
        else:
            error+= len(pred)-len(gold)
            break

    accuracy = (total - error) / total * 100
    return error, total, accuracy

            
    
# MAIN
train_file = sys.argv[1]
test_file = sys.argv[2]
if not os.path.exists(train_file):
    print("CANNOT FIND " + train_file)
    
if not os.path.exists(test_file):
    print("CANNOT FIND " + test_file)
    

tr_raw_x = [] 
tr_raw_y = [] 
all_tr_data = ""
tr_max_len = 0 
with open(train_file, 'r') as f_names:
    for name in f_names.readlines():
        print("Getting data from " + name) 
        name = name.strip()
        r_x, r_y, data, mx_len = get_data(name)
        tr_raw_x += r_x
        tr_raw_y += r_y
        all_tr_data += data
        if tr_max_len < mx_len:
            tr_max_len = mx_len


test_raw_x, test_raw_y, all_test_data, test_max_len = get_data(test_file)

if test_max_len > tr_max_len:
    raise ValueError('THERE IS A LINE IN THE TEST DATA THAT IS LONGER THAN THE LINES IN THE TRAINING DATA. REFORMAT THE TEST DATA SUCH THAT THE LINES ARE NO LONGER THAN', tr_max_len)

idx_vocab, vocab_idx, tr_vocab_size = make_train_dict(all_tr_data)
vocab_tuple = (idx_vocab, vocab_idx, tr_vocab_size)

MAX_LEN = tr_max_len
EDIT_SPACE= 5

tr_X_data = [[vocab_idx[c] for c in arr] for arr in tr_raw_x]
tr_Y_data = [[vocab_idx[c] for c in arr] for arr in tr_raw_y]

tr_X_data = np.array([np.pad(line, (0, MAX_LEN-len(line)), 'constant', constant_values=0) for line in tr_X_data])
tr_Y_data = np.array([np.pad(line, (0, MAX_LEN-len(line)+EDIT_SPACE), 'constant', constant_values=0) for line in tr_Y_data])

test_X_data = [[vocab_idx[c] for c in arr] for arr in test_raw_x]
test_Y_data = [[vocab_idx[c] for c in arr] for arr in test_raw_y]

test_X_data = np.array([np.pad(line, (0, MAX_LEN-len(line)), 'constant', constant_values=0) for line in test_X_data])
test_Y_data = np.array([np.pad(line, (0, MAX_LEN-len(line)+EDIT_SPACE), 'constant', constant_values=0) for line in test_Y_data])

EPOCHS = 1000
NUM_IN_BATCH = 100
CHECKPOINT = "pretrained/bidirectENC_dynamDEC/5-25-2018"

with open(CHECKPOINT+"-vocab-dictionaries", 'w') as jsonfile:
    json.dump(vocab_tuple, jsonfile)

t = time.time()
graph = build_multilayer_lstm_graph_with_dynamic_rnn(num_classes=tr_vocab_size)
print("It took", time.time() - t, "seconds to build the graph...")

print('Training...')
t = time.time()
train_network(graph, CHECKPOINT)
print("It took", time.time() - t, "seconds to train the model...")

print("Evaluating test data...")
chars = eval_network(graph, CHECKPOINT, idx_vocab)
err, total_chars, acc = char_err_rate(chars, test_raw_y)
print(chars)

print("Accuracy: ", acc)
print("Total errors: ", err)
print("Total chars: ", total_chars)
