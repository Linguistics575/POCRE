# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:01:46 2018

@author: aymm
Based on tutorial in https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
"""

import numpy as np
import tensorflow as tf
#matplotlib inline
#import matplotlib.pyplot as plt
import time
import os
import sys
#import urllib.request
#from tensorflow.models.rnn.ptb import reader

"""
Load and process data, utility functions
"""
        
def get_data(file_name):
    with open(file_name,'r') as f:
        count = 1
        raw_x = str()
        raw_y = str()
        for line in f.readlines():
            if count % 3 == 1:
                raw_y += line.strip() + '\n'
            elif count % 3 == 2:
                raw_x += line.strip() + '\n'
            count += 1
        
        print("X Training Data Length:", len(raw_x))
        print("Y Training Data length:", len(raw_y))

    return raw_x, raw_y
        
        
def make_train_dict(raw_x, raw_y):
    x_vocab = set(raw_x)
    y_vocab = set(raw_y)
    vocab = x_vocab.union(y_vocab)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    
    return idx_to_vocab, vocab_to_idx, vocab_size
    
    
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()    
    
    
def build_multilayer_lstm_graph_with_dynamic_rnn(
    num_classes, 
    state_size = 100,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()
    
    batch_size = tf.placeholder(tf.int64)

    x = tf.placeholder(tf.int32, shape=[None, 1], name='input_placeholder')
    y = tf.placeholder(tf.int32, shape=[None, 1], name='labels_placeholder')

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
    
    data_iter = dataset.make_initializable_iterator()
    features, labels = data_iter.get_next()
    
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, features)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    """
    Test dropout
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=global_dropout)
    """
    dynam_batch_size = tf.shape(features)[0]
    init_state = cell.zero_state(dynam_batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(labels, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    # Different optimizer recommended by https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        data_iter = data_iter,
        batch_size = batch_size,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )
    
    

        
"""
CURRENTLY BATCHES ARE A WINDOW OF 20, FOR SLIDING WINDOW GEN_EPOCHS NEEDS TO CHANGE
"""
def train_network(g, verbose=True, save=False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        sess.run(g['data_iter'].initializer, feed_dict={g['x']: tr_X_data, g['y']: tr_Y_data, 
                                          g['batch_size']: BATCH_SIZE})
                                          
        training_losses = []
        for i in range(EPOCHS):
            steps = 0
            tot_loss = 0
            for _ in range(N_BATCHES):
                _, loss_value = sess.run([g['train_step'],g['total_loss']])
                tot_loss += loss_value
                steps += 1

            if verbose:
                print("Average training loss for Epoch", i, ":", tot_loss/steps)
            training_losses.append(tot_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses
    

    
def eval_network(g):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #predictions = []
        #gold_standard = []
        sess.run(g['data_iter'].initializer, feed_dict={g['x']: test_X_data, 
                                g['y']: test_Y_data, g['batch_size']: len(test_X_data)})
        print(sess.run([g['total_loss']]))         
    


def char_err_rate(pred, gold):
    total = 0.
    error = 0.
    for i in range(len(pred)):
        total+= 1
        if not pred[i] == gold[i]:
            error+= 1
            
    accuracy = error / total
    return error, total, accuracy
            
    

train_file = sys.argv[1]
test_file = sys.argv[2]
if not os.path.exists(train_file):
    print("CANNOT FIND " + train_file)
    
if not os.path.exists(test_file):
    print("CANNOT FIND " + test_file)
    

tr_raw_x, tr_raw_y = get_data(train_file)
test_raw_x, test_raw_y = get_data(test_file)

idx_vocab, vocab_idx, tr_vocab_size = make_train_dict(tr_raw_x, tr_raw_y)

tr_X_data = np.vstack([vocab_idx[c] for c in tr_raw_x])
tr_Y_data = np.vstack([vocab_idx[c] for c in tr_raw_y])

test_X_data = np.vstack([vocab_idx[c] for c in test_raw_x])
test_Y_data = np.vstack([vocab_idx[c] for c in test_raw_y])

EPOCHS = 10
BATCH_SIZE = 20
N_BATCHES = len(tr_X_data) // BATCH_SIZE
t = time.time()
graph = build_multilayer_lstm_graph_with_dynamic_rnn(num_classes=tr_vocab_size)
print("It took", time.time() - t, "seconds to build the graph...")

print('Training...')
train_network(graph)

print("Evaluating test data...")
eval_network(graph)

"""
convert_pred = [idx_vocab[c] for c in pred]
convert_gold = [idx_vocab[c] for c in gold]

print(convert_pred)
print(convert_gold)

print("Character error rate:", char_err_rate(convert_pred, convert_gold))
"""
