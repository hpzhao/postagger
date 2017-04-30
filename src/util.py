#!/usr/bin/env python
#coding:utf8
from copy import deepcopy
import numpy
import tensorflow as tf
numpy.random.seed(1)
def get_context(batch_x):
    batch_x_with_context = numpy.zeros(shape=(batch_x.shape[0],batch_x.shape[1],3))
    batch_x_copy = deepcopy(batch_x)
    for x_index in range(batch_x_copy.shape[0]):
        x = batch_x_copy[x_index]
        for word_index in range(x.shape[0]):
            if word_index == 0:
                batch_x_with_context[x_index][word_index][0] = 0
            else:
                batch_x_with_context[x_index][word_index][0] = x[word_index-1]
            batch_x_with_context[x_index][word_index][1] = x[word_index]
            if word_index == x.shape[0] - 1:
                batch_x_with_context[x_index][word_index][2] = 0
            else:
                batch_x_with_context[x_index][word_index][2] = x[word_index + 1]

    return batch_x_with_context

def gpu_config():
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config
def padding_batch(data):
    assert isinstance(data,numpy.ndarray),'the data type should be numpy.ndarray whose dtype is list'

    max_seg = max([len(instance) for instance in data])
    batch_size = data.shape[0]
    batch = numpy.zeros([batch_size, max_seg],numpy.int32)
    mask = numpy.zeros([batch_size, max_seg],numpy.int32)
    for i in range(batch_size):
        cur_len = numpy.sign(data[i]).sum()
        zeros = [0 for _ in range(max_seg - cur_len)]
        ones = [1 for _ in range(cur_len)]
        batch_i = deepcopy(data[i])
        if cur_len == 0:
            batch[i] = zeros
            mask[i] = zeros
            continue
        if cur_len < max_seg:
            batch_i.extend(zeros)
            ones.extend(zeros)
         
        batch[i] = batch_i
        mask[i] = ones

    return batch,mask
    
def padding(data,labels):

    assert isinstance(data,numpy.ndarray),'the data type should be numpy.ndarray whose dtype is list'
    assert isinstance(labels,numpy.ndarray),'the label type should be numpy.ndarray whose dtype is list'
    assert data.shape == labels.shape,'data shape and labels shape should be same'

    max_seg = max([len(instance) for instance in data])
    batch_size = data.shape[0]
    batch_x = numpy.zeros([batch_size, max_seg],numpy.int32)
    batch_y = numpy.zeros([batch_size, max_seg],numpy.int32)
    mask = numpy.zeros([batch_size, max_seg],numpy.int32)

    for i in range(batch_size):
        cur_len = numpy.sign(data[i]).sum()
        zeros = [0 for _ in range(max_seg - cur_len)]
        ones = [1 for _ in range(cur_len)]
        batch_x_i = deepcopy(data[i])
        batch_y_i = deepcopy(labels[i])
        if cur_len == 0:
            batch_x[i] = zeros
            batch_y[i] = zeros
            mask[i] = zeros
            continue
        if cur_len < max_seg:
            batch_x_i.extend(zeros)
            batch_y_i.extend(zeros)
            ones.extend(zeros)
         
        batch_x[i] = batch_x_i
        batch_y[i] = batch_y_i
        mask[i] = ones
    batch_y = batch_y.reshape([batch_size * max_seg])
    mask = mask.reshape([batch_size * max_seg])
    return batch_x,batch_y,mask
