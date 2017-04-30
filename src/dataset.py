#!/usr/bin/env python
#coding:utf8
import numpy
numpy.random.seed(1)
class Dataset():
    def __init__(self,data_list,labels_list):
        self._data_list = data_list
        self._labels_list = labels_list
        self._data = numpy.array(data_list)
        self._labels = numpy.array(labels_list)
        self._numbers = self._data.shape[0]
        self._index_in_epoch = 0
        self._epochs = 0
        self._perm = None
    
    def reset(self):
        self._index_in_epoch = 0
        self._epochs = 0
    def numbers(self):
        return self._numbers
    def epochs(self):
        return self._epochs
    def data(self):
        return self._data
    def labels(self):
        return self._labels
    def perm(self):
        return self._perm
    def fake_data(self,batch_size):
        assert batch_size <= self._numbers,'batch size should not be bigger than numbers'
        pad_num = (batch_size - self._numbers % batch_size) if self._numbers % batch_size != 0 else 0
        pad_data = [0]
        for _ in range(pad_num):
            self._data_list.append(pad_data)
            self._labels_list.append(pad_data)
        
        self._data = numpy.array(self._data_list)
        self._labels = numpy.array(self._labels_list)
        self._numbers = self._data.shape[0]
        self._index_in_epoch = 0
        self._epochs = 0

    def all(self):
        return self._data,self._labels
    def next_batch(self,batch_size,perm = None):
        if perm is None:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._numbers:
                #finished epoch
                self._epochs += 1
                #shuffle data
                self._perm = numpy.arange(self._numbers)
                numpy.random.shuffle(self._perm)
                self._data = self._data[self._perm]
                self._labels = self._labels[self._perm]
                #start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._numbers
            end = self._index_in_epoch
            return self._data[start:end],self._labels[start:end]
        else:
            self._epochs += 1
            #shuffle data
            self._perm = perm 
            self._data = self._data[self._perm]
            self._labels = self._labels[self._perm]
            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._numbers
            end = self._index_in_epoch
            return self._data[start:end],self._labels[start:end]

if __name__ == '__main__':
    data = [[1,2],[3,4],[5,6]]
    labels = [[1,1],[2,2],[3,3]]
    test = Dataset(data,labels)
    print test.numbers()
    print test.data()
    print test.labels()
    test.fake_data(2)
    print test.numbers()
    print test.data()
    print test.labels()
    print test.next_batch(2)[0] 
    print test.next_batch(2)[0] 
    print test.next_batch(2)[0] 
    print test.next_batch(2)[0] 
