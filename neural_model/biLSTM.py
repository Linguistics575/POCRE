# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:01:46 2018

@author: aymm
Based on tutorial in https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
"""

import numpy as np
import tensorflow as tf
#matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import sys
import urllib.request
from tensorflow.models.rnn.ptb import reader

"""
Load and process data, utility functions
"""
        
def get_data(file_name):
    with open(file_name,'r') as f:
        raw_data = f.read()
        print("Data length:", len(raw_data))
    
    
    # TODO: CHANGE TO CHARACTER ALIGNMENT
    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    
    data = [vocab_to_idx[c] for c in raw_data]
    del raw_data
    return X, Y, vocab_size
        

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()    
    
    
def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = 100,
    num_classes = tr_vocab_size,
    batch_size = 32,
    num_steps = 200,
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
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
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
def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        raw_data = np.array(tr_data, dtype=np.int32)

        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
            
        epoch_size = (batch_len - 1) // num_steps
    
        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)
        

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
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
    
tr_X_data, tr_Y_data, tr_vocab_size = get_data(train_file)

t = time.time()
graph = build_multilayer_lstm_graph_with_dynamic_rnn()
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

