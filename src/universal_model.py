#!/usr/bin/env python
#coding:utf8
from __future__ import division
from copy import deepcopy

import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format = '%(asctime)s [%(levelname)s] %(message)s')

class Char_Based_BiLSTM():

    def __init__(self,char_size,word_size,epochs,word_embedding,pos_classes = 17):
        # Parameters
        self.pos_classes = pos_classes

        self.cluster_dim = 50
        self.char_dim = 50
        self.word_dim = 100
        
        self.cluster_size = 257
        self.char_size = char_size
        self.word_size = word_size
        
        self.char_hidden = 100
        self.word_hidden = 300
        
        self.learning_rate = 1e-3
        self.dropout = 0.5
        self.l2 = 1e-3
        self.epochs = epochs
        
        # set random seed
        tf.set_random_seed(1)
        
        # embedding
        with tf.variable_scope('embedding'):
            self.char_embedding = tf.get_variable('char_embedding',[self.char_size,self.char_dim])
            self.word_embedding = tf.Variable(initial_value = word_embedding,trainable = False,dtype = tf.float32)
            self.cluster_embedding = tf.get_variable('cluster_embedding',[self.cluster_size,self.cluster_dim],tf.float32)
        # define placeholder
        with tf.variable_scope('placeholder'):
            self.cluster_input = tf.placeholder(tf.int32,[1,None])
            self.word_input = tf.placeholder(tf.int32,[1,None])
            self.char_input = tf.placeholder(tf.int32,[None])
            self.char_shape = tf.placeholder(tf.int32,[2])
            self.targets = tf.placeholder(tf.int32,[None])
            self.output_keep_prob = tf.placeholder(tf.float32)
        # build char_based bilstm graph
        with tf.variable_scope('char_based_bilstm'):
            self.char_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden)
            self.char_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.char_hidden)
            
            self.char_input_maxtrix = tf.reshape(self.char_input,shape = self.char_shape)
            self.char_seq_len = tf.reduce_sum(tf.sign(self.char_input_maxtrix),reduction_indices = 1)
            self.char_seq_len = tf.cast(self.char_seq_len,tf.int32)

            self.inputs = tf.nn.embedding_lookup(self.char_embedding,self.char_input_maxtrix)
            self.char_outputs,_ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = self.char_cell_fw,
                    cell_bw = self.char_cell_bw,
                    inputs = self.inputs,
                    sequence_length = self.char_seq_len,
                    dtype=tf.float32)

            # get the last valid outputs
            self.outputs_fw,self.outputs_bw = self.char_outputs
            self.batch_range = tf.range(tf.shape(self.char_seq_len)[0])
            self.fw_indices = tf.stack([self.batch_range,self.char_seq_len - 1],axis = 1)
            self.zeros = tf.zeros([tf.shape(self.char_seq_len)[0]],dtype = tf.int32)
            self.bw_indices = tf.stack([self.batch_range,self.zeros],axis = 1)
            
            self.last_fw_outputs = tf.gather_nd(self.outputs_fw,self.fw_indices)
            self.last_bw_outputs = tf.gather_nd(self.outputs_bw,self.bw_indices)

            self.last_outputs = tf.concat(1,[self.last_fw_outputs,self.last_bw_outputs])
            self.x = tf.reshape(self.last_outputs,[1,-1,2 * self.char_hidden])

        with tf.variable_scope('word_based_bilstm_input_layer'):
            self.word_inputs = tf.nn.embedding_lookup(self.word_embedding,self.word_input)
            self.cluster_inputs = tf.nn.embedding_lookup(self.cluster_embedding,self.cluster_input)
            self.x = tf.concat(2,[self.word_inputs,self.x,self.cluster_inputs])
            
            self.input_dim = 2 * self.char_hidden + self.word_dim + self.cluster_dim
            self.x = tf.reshape(self.x,[-1,self.input_dim])
            self.input_w = tf.get_variable('input_w',[self.input_dim,self.word_dim])
            self.input_b = tf.get_variable('input_b',[self.word_dim])
            self.x = tf.nn.relu(tf.matmul(self.x,self.input_w) + self.input_b)
            self.x = tf.reshape(self.x,[1,-1,self.word_dim])
        # build the word based bilstm graph  
        with tf.variable_scope('word_based_bilstm'):
            self.word_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.word_hidden)
            self.word_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.word_hidden)
            
            self.word_seq_len = tf.reduce_sum(tf.sign(self.word_input),reduction_indices = 1)
            self.word_seq_len = tf.cast(self.word_seq_len,tf.int32)

            self.word_outputs,_ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = self.word_cell_fw,
                    cell_bw = self.word_cell_bw,
                    inputs = self.x,
                    sequence_length = self.word_seq_len,
                    dtype = tf.float32)
            self.outputs = tf.concat(2,self.word_outputs)
            self.outputs = tf.reshape(self.outputs,[-1,2 * self.word_hidden])
            self.outputs = tf.nn.dropout(self.outputs,keep_prob = self.output_keep_prob)

        with tf.variable_scope('logits_layer'):
            self.logits_w = tf.get_variable("logits_w",[2 * self.word_hidden,self.pos_classes],tf.float32)
            self.logits_b = tf.get_variable("logits_b",[self.pos_classes],tf.float32)
            self.logits = tf.matmul(self.outputs, self.logits_w) + self.logits_b
        
        with tf.variable_scope('train'):
            self.l2_loss = self.l2 * tf.nn.l2_loss(self.logits_w)
            self.softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,self.targets)
            self.loss = tf.reduce_mean(self.softmax)  + self.l2_loss
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        with tf.variable_scope('eval'):
            self.correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(self.logits,self.targets,1),tf.int32))
            _,self.predict = tf.nn.top_k(self.logits)
    
    def padding(self,data):
        assert isinstance(data,np.ndarray)
        max_seg = max([len(instance) for instance in data])
        batch_size = data.shape[0]
        batch = np.zeros([batch_size, max_seg],np.int32)
        mask = np.zeros([batch_size, max_seg],np.int32)
        for i in range(batch_size):
            cur_len = np.sign(data[i]).sum()
            zeros = [0 for _ in range(max_seg - cur_len)]
            ones = [1 for _ in range(cur_len)]
            batch_i = deepcopy(data[i])
            
            if cur_len < max_seg:
                batch_i.extend(zeros)
                ones.extend(zeros)
             
            batch[i] = batch_i
            mask[i] = ones
        batch = np.reshape(batch,[-1])
        return batch,mask
    def test(self,sess,dev_data):
        tokens,corrects = (0,0)
        predicts = [] 
        dev_word,dev_char,dev_cluster = dev_data
        for _ in range(dev_word.numbers()):
            word_input,word_target = dev_word.next_batch(1)
            word_input = np.reshape(np.array(word_input[0]),[1,-1])
            word_target = np.reshape(np.array(word_target[0]),[-1])
            char_input,_ = dev_char.next_batch(1)
            cluster_input,_ = dev_cluster.next_batch(1)
            cluster_input = np.reshape(np.array(cluster_input[0]),[1,-1])
            char_input,char_mask = self.padding(np.array(char_input[0]))
            char_shape = char_mask.shape
            feed_dict = {self.cluster_input:cluster_input,self.word_input:word_input,self.char_input:char_input,
                    self.targets:word_target,self.output_keep_prob:1,self.char_shape:char_shape}
            correct,predict = sess.run([self.correct,self.predict],feed_dict = feed_dict)
            tokens += np.sign(word_input).sum()
            corrects += correct
            predicts.extend(np.reshape(predict,[-1]))

        dev_word.reset()
        dev_char.reset()
        dev_cluster.reset()
        return corrects / tokens, predicts
    def train(self,sess,train_data,dev_data,sava_path):
        saver = tf.train.Saver()
        
        train_word,train_char,train_cluster = train_data
         
        data_size = train_word.numbers()
        train_epoch = 0
        best_acc = 0
        best_predict = []
        for step in range(data_size * self.epochs):
            word_input,word_target = train_word.next_batch(1)
            word_input = np.reshape(np.array(word_input[0]),[1,-1])
            word_target = np.reshape(np.array(word_target[0]),[-1])
            # make word_data char_data cluster_data shuffle in a same perm
            if train_word.epochs() > train_epoch:
                train_epoch += 1
                perm = train_word.perm()
                char_input,_ = train_char.next_batch(1,perm)
                cluster_input,_ = train_cluster.next_batch(1,perm)
            else:
                char_input,_ = train_char.next_batch(1)
                cluster_input,_ = train_cluster.next_batch(1)
            cluster_input = np.reshape(np.array(cluster_input[0]),[1,-1])
            char_input,char_mask = self.padding(np.array(char_input[0]))
            char_shape = char_mask.shape
            feed_dict = {self.cluster_input:cluster_input,self.word_input:word_input,self.char_input:char_input,
                    self.targets:word_target,self.output_keep_prob:1-self.dropout,self.char_shape:char_shape}

            sess.run(self.train_op,feed_dict=feed_dict)
            
            if step % 100 == 0:
                cur_acc , cur_predict = self.test(sess,dev_data)
                logging.info('epochs:' + str(train_epoch) + ' accuracy:' + str(cur_acc))
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    best_predict = cur_predict
                    saver.save(sess,sava_path)
        return best_acc,best_predict
