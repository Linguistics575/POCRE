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
        raw_init = str()
        for line in f.readlines():
            if not count == 1:
                if count % 4 == 2:
                    raw_y += line.strip() + '\n'
                elif count % 4 == 3:
                    raw_x += line.strip() + '\n'
                elif count % 4 == 0:
                    raw_init += line
            count += 1
        print("Data length:", len(raw_y))

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
    batch_size = 20, # number of characters in each input
    num_steps = 10,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    """
    Test dropout
    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=global_dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=global_dropout)
    """
    
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    # Different optimizer recommended by https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )
    
    
"""
 original code https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/rnn/ptb/reader.py
Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
def data_iterator(batch_size, num_steps):
    raw_X_data = np.array(tr_X_data, dtype=np.int32)
    raw_Y_data = np.array(tr_Y_data, dtype=np.int32)

    data_len = len(raw_X_data)
    batch_len = data_len // batch_size
    x_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    y_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        start_ind = batch_len * i
        end_ind = batch_len * (i + 1)
        x_data[i] = raw_X_data[start_ind:end_ind]
        y_data[i] = raw_Y_data[start_ind:end_ind]
            
    epoch_size = (batch_len - 1) // num_steps
    
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = x_data[:, i*num_steps:(i+1)*num_steps]
        y = y_data[:, i*num_steps:(i+1)*num_steps]
        yield (x, y)
    
    
def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield data_iterator(batch_size, num_steps)
        
"""
CURRENTLY BATCHES ARE A WINDOW OF 20, FOR SLIDING WINDOW GEN_EPOCHS NEEDS TO CHANGE
"""
def train_network(g, num_epochs, num_steps = 10, batch_size = 20, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses
    



train_file = sys.argv[1]
test_file = sys.argv[2]
if not os.path.exists(train_file):
    print("CANNOT FIND " + train_file)
    
if not os.path.exists(test_file):
    print("CANNOT FIND " + test_file)
    

tr_raw_x, tr_raw_y = get_data(train_file)
test_raw_x, test_raw_y = get_data(test_file)

idx_vocab, vocab_idx, tr_vocab_size = make_train_dict(tr_raw_x, tr_raw_y)

tr_X_data = [vocab_idx[c] for c in tr_raw_x]
tr_Y_data = [vocab_idx[c] for c in tr_raw_y]

test_X_data = [vocab_idx[c] for c in test_raw_x]
test_Y_data = [vocab_idx[c] for c in test_raw_y]


t = time.time()
graph = build_multilayer_lstm_graph_with_dynamic_rnn(num_classes=tr_vocab_size)
print("It took", time.time() - t, "seconds to build the graph.")
t = time.time()
train_network(graph, 3)
print("It took", time.time() - t, "seconds to train for 3 epochs.")

"""
feed_dict={g['x']: X, g['y']: Y}

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

print classification
"""

