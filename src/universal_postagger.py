#!/usr/bin/env python
#coding:utf8
from dataset import Dataset
from universal_model import Char_Based_BiLSTM
import cPickle as pkl
import tensorflow as tf
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO,format = '%(asctime)s [%(levelname)s] %(message)s')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('is_training',1,'is training or testing,default is training')
flags.DEFINE_integer('epochs',10,'epochs of train')
flags.DEFINE_string('model','./best.model','model used in test')
flags.DEFINE_string('language','en','language')
flags.DEFINE_string('pos_type','upos','type of pos')

flags.DEFINE_string('emb_path','./','path of embedding')
flags.DEFINE_string('cluster_path','./','path of brown cluster')
flags.DEFINE_string('data_path','./','path of data')
flags.DEFINE_string('train_file','train.conll','filename of train data')
flags.DEFINE_string('dev_file','dev.conll','filename of dev data')
flags.DEFINE_string('output','-ppos.conll','output filename')


upos2id = {}
id2upos = {}
xpos2id = {}
id2xpos = {}
word2id = {}
char2id = {}
id2char = {}
word_list = []
word2cluster = {}
char_size = 0
word_size = 0
xpos_size = 0
embedding = None

def gen_pos_map(train_file):
    global upos2id,id2upos,xpos2id,id2xpos,word_list,xpos_size

    upos = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM',
            'PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']

    for id,pos in enumerate(upos):
        id2upos[id] = pos
        upos2id[pos] = id

    xpos = set()
    word_set = set()
    for line in open(train_file):
        line = line.strip().decode('utf8')
        if line and line[0] != u'#':
            tokens = line.split('\t')
            if u'-' not in tokens[0] and u'.' not in tokens[0]:
                xpos.add(tokens[4])
                word_set.add(tokens[1])
    
    for id,pos in enumerate(xpos):
        id2xpos[id] = pos
        xpos2id[pos] = id
    xpos_size = len(xpos2id)
    word_list = list(word_set)

def gen_char_map():
    global char2id,id2char,word_list,char_size
    char_set = set()
    for word in word_list:
        for char in word:
            char_set.add(char)
    char_list = list(char_set)

    # 保证打乱训练数据也得到唯一字符序列
    char_list.sort()

    for id,char in enumerate(char_list):
        char2id[char] = id + 2
        id2char[id + 2] = char

    char_size = len(char2id) + 2

def gen_word_map(embedding_file):

    
    global word2id,embedding,word_size
    
    # 第0维为padding预留
    # 第1维为UNK预留
    
    emb_list = []
    zeros = [0 for _ in range(100)]
    emb_list.append(zeros)
    emb_list.append(zeros)

    word_set = set()
    id = 2

    for line in open(embedding_file):
        line = line.strip().decode('utf8')
        if line:
            tokens = line.split()
            if len(tokens) != 2 and tokens[0] != u'</s>':
                assert len(tokens) == 101, "the embedding dim must be 100"
                #主要处理英语大小写导致太多UNK，对其他语言没影响
                word = tokens[0].lower()
                if word not in word2id:
                    emb_list.append(tokens[1:])
                    word2id[word] = id
                    id += 1
    #embedding = emb
    embedding = np.array(emb_list,np.float32)
    word_size = embedding.shape[0]

def gen_cluster_map(cluster_file):
    global word2cluster
    cluster2id = {}
    for line in open(cluster_file):
        line = line.strip()
        if line:
            tokens = line.decode('utf8').split('\t')
            if tokens[0] in cluster2id:
                word2cluster[tokens[1]] = cluster2id[tokens[0]]
            else:
                word2cluster[tokens[1]] = len(cluster2id) + 1
                cluster2id[tokens[0]] = len(cluster2id) + 1

def preprocess():
    logging.info('start preprocess')
    gen_pos_map(os.path.join(FLAGS.data_path,FLAGS.train_file))
    gen_char_map() 
    gen_word_map(FLAGS.emb_path)
    gen_cluster_map(FLAGS.cluster_path)
    logging.info('end preprocess')

def build_dataset(data_file):
    global word2id,char2id,word2cluster,upos2id,xpos2id

    sent_words_list = []
    sent_chars_list = []
    sent_clusters_list = []
    sent_upos_list = []
    sent_xpos_list = []
    
    words_list = []
    chars_list = []
    clusters_list = []
    upos_list = []
    xpos_list = []

    for line in open(data_file):
        line = line.strip().decode('utf8')
        if line and line[0] != u'#':
            tokens = line.split('\t')
            if u'-' not in tokens[0] and u'.' not in tokens[0]:
                word = tokens[1].lower()
                words_list.append(word2id[word] if word in word2id else 1)
                chars_list.append([char2id[char] if char in char2id else 1 for char in word])
                clusters_list.append(word2cluster[word] if word in word2cluster else 0)
                upos,xpos = tokens[3:5]
                upos_list.append(upos2id[upos] if upos in upos2id else 0)
                xpos_list.append(xpos2id[xpos] if xpos in xpos2id else 0)
        if line == '':
            sent_words_list.append(words_list)
            sent_chars_list.append(chars_list)
            sent_clusters_list.append(clusters_list)
            sent_xpos_list.append(xpos_list)
            sent_upos_list.append(upos_list)
            
            words_list = []
            chars_list = []
            clusters_list = []
            upos_list = []
            xpos_list = []
    upos_word_dataset = Dataset(sent_words_list,sent_upos_list)
    xpos_word_dataset = Dataset(sent_words_list,sent_xpos_list)
    char_dataset = Dataset(sent_chars_list,sent_upos_list)
    cluster_dataset = Dataset(sent_clusters_list,sent_upos_list)
    return upos_word_dataset,xpos_word_dataset,char_dataset,cluster_dataset
def write(input,output,result,column,pos_map):
    index = 0
    with open(output,'w') as f:
        for line in open(input):
            line = line.strip().decode('utf8')
            if line and line[0] != u'#':
                tokens = line.split('\t')
                if '.' in tokens[0]:
                    continue
                elif u'-' not in tokens[0]:
                    tokens[column] = pos_map[result[index]]
                    index += 1
                f.write('\t'.join(tokens).encode('utf8') + '\n')
            else:
                f.write(line.encode('utf8') + '\n')
    
def train():
    global id2upos,xpos_size,word_size,char_size,embedding

    logging.info('start training...')
    logging.info('start building dataset')
    train_file = os.path.join(FLAGS.data_path,FLAGS.train_file)
    train_upos_word_dataset,train_xpos_word_dataset,train_char_dataset,train_cluster_dataset = build_dataset(train_file)
    dev_file = os.path.join(FLAGS.data_path,FLAGS.dev_file)
    dev_upos_word_dataset,dev_xpos_word_dataset,dev_char_dataset,dev_cluster_dataset = build_dataset(dev_file)
    logging.info('end building dataset')
    if FLAGS.pos_type == 'upos': 
        prefix = FLAGS.language
        save_path = os.path.join(FLAGS.data_path,prefix + '.upos.model')
        with tf.variable_scope('model'):
            model = Char_Based_BiLSTM(char_size,word_size,epochs = FLAGS.epochs,word_embedding = embedding,pos_classes = 17)
        
        config = tf.ConfigProto()
        config.inter_op_parallelism_threads = 2
        config.intra_op_parallelism_threads = 1
        with tf.Session(config=config) as sess:
            train_data = (train_upos_word_dataset,train_char_dataset,train_cluster_dataset)
            dev_data = (dev_upos_word_dataset,dev_char_dataset,dev_cluster_dataset)
            sess.run(tf.global_variables_initializer())
            acc,predict = model.train(sess,train_data,dev_data,save_path)         
            logging.info('best accuracy:' + str(acc))
            output_name = FLAGS.dev_file.split('.')[0] + FLAGS.output
            output_file = os.path.join(FLAGS.data_path,output_name)
            write(dev_file,output_file,predict,3,id2upos)
            logging.info('output file:' + output_file)
    logging.info('end training')

def test():
    global xpos_size,word_size,char_size,embedding
    logging.info('start testing....')
    logging.info('start building dataset') 
    dev_file = os.path.join(FLAGS.data_path,FLAGS.dev_file)
    dev_upos_word_dataset,dev_xpos_word_dataset,dev_char_dataset,dev_cluster_dataset = build_dataset(dev_file)
    logging.info('end building dataset')

    if FLAGS.pos_type == 'upos': 
        with tf.variable_scope('model'):
            model = Char_Based_BiLSTM(char_size,word_size,epochs = FLAGS.epochs,word_embedding = embedding,pos_classes = 17)
        
        with tf.Session() as sess:
            dev_data = (dev_upos_word_dataset,dev_char_dataset,dev_cluster_dataset)
            saver = tf.train.Saver()
            model_path = os.path.join(FLAGS.data_path,FLAGS.model)
            saver.restore(sess,model_path)
            acc,predict = model.test(sess,dev_data)
            logging.info('best accuracy:' + str(acc))
            output_name = FLAGS.dev_file.split('.')[0] + FLAGS.output
            output_file = os.path.join(FLAGS.data_path,output_name)
            write(dev_file,output_file,predict,3,id2upos)
            logging.info('output file:' + output_file)
    logging.info('end testing')
def main(_):
    preprocess()
    if FLAGS.is_training == 1:
        train()
    else:
        test()
if __name__ == '__main__':
    tf.app.run()
