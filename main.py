# -*- coding: utf-8 -*-
import os
import pickle
import random
import argparse
import datetime

import numpy
import torch

import model
import train
import datasetswb

random.seed(66)
torch.manual_seed(66)

parser = argparse.ArgumentParser(description='classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=128)
parser.add_argument('-batch-size', type=int, default=16)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=100)
parser.add_argument('-save-interval', type=int, default=100)
parser.add_argument('-save-dir', type=str, default='snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=True)
# model
parser.add_argument('-dropout-embed', type=float, default=0.5)
parser.add_argument('-dropout-rnn', type=float, default=0.3)

parser.add_argument('-use-embedding', action='store_true', default=True)
parser.add_argument('-max-norm', type=float, default=None)
parser.add_argument('-embed-dim', type=int, default=300)

parser.add_argument('-input-size', type=int, default=300)
parser.add_argument('-hidden-size', type=int, default=200)

parser.add_argument('-kernel-num', type=int, default=200)
parser.add_argument('-kernel-sizes', type=str, default='3')
parser.add_argument('-static', action='store_true', default=False)

parser.add_argument('-which-model', type=str, default='mybilstm')
# device
parser.add_argument('-device', type=int, default=-1)
parser.add_argument('-no-cuda', action='store_true', default=True)
# option
parser.add_argument('-snapshot', type=str, default=None)
parser.add_argument('-predict', type=str, default=None)
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-label-num', type=int, default=2)
parser.add_argument('-lr-scheduler', type=str, default=None)
parser.add_argument('-clip-norm', type=str, default=None)

args = parser.parse_args()


# create embedding
def getEmbedding(plk_path, embed_path, id2word, name):

    # 如果这个词向量已经保存为pkl了，就直接加载
    if os.path.exists(plk_path):
        plk_f = open(plk_path, 'rb+')
        m_embed = pickle.load(plk_f)
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)
        plk_f.close()
    else:
        # 如果没有找到pkl，就从已经筛选好的词向量里去找
        assert os.path.exists(embed_path)
        embed_f = open(embed_path, encoding="utf-8")
        m_dict = {}
        for idx, line in enumerate(embed_f.readlines()):
            if not line == '':
                strs = line.split(' ')
                m_dict[strs[0]] = [float(i) for idx2, i in enumerate(strs) if not idx2 == 0]
        embed_f.close()

        m_embed = []
        notfound = 0
        for idx in range(len(id2word)):
            if id2word[idx] in m_dict:
                m_embed.append(m_dict[id2word[idx]])
            else:
                notfound += 1
                m_embed.append([round(random.uniform(-0.25, 0.25), 6) for i in range(args.embed_dim)])
        print('notfound:', notfound)
        print('ratio:', notfound / len(id2word))
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)

        f = open(os.path.join('./data', name), 'wb+')
        # pickle.dump(id2word, f)
        pickle.dump(m_embed, f)
        f.close()
    return m_embedding


# load data
print("\nLoading data...")
'''not understand'''
train_data = datasetswb.splitcorpus("./data/raw.clean.train", args.label_num, args.shuffle)
dev_data = datasetswb.splitcorpus("./data/raw.clean.dev", args.label_num, args.shuffle)
test_data = datasetswb.splitcorpus("./data/raw.clean.test", args.label_num, args.shuffle)

vocabulary_text = datasetswb.Vocabulary.makeVocabularyByText([train_data])
vocabulary_label = datasetswb.Vocabulary.makeVocabularyByLable([train_data])

train_iter = datasetswb.MyIterator(args.batch_size, train_data, vocabulary_text, vocabulary_label).iterators
dev_iter = datasetswb.MyIterator(len(dev_data), dev_data, vocabulary_text, vocabulary_label).iterators
test_iter = datasetswb.MyIterator(len(test_data), test_data, vocabulary_text, vocabulary_label).iterators


# load embedding
m_embedding = None
if args.use_embedding:
    id2word = vocabulary_text.id2word
    # m_embedding = getEmbedding('./data/conj300d.pkl',
    #                            './data/glove.sentiment.conj.pretrained.txt',
    #                            id2word,
    #                            'conj300d.pkl')
    m_embedding = getEmbedding('./data/840b300d.pkl',
                               'D:/AI/embedding&corpus/glove300d.txt',
                               id2word,
                               '840b300d.pkl')


# update args and print
args.embed_num = len(vocabulary_text.word2id)
args.class_num = args.label_num
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
m_model = None
rtrain = True
if args.snapshot is None:
    if args.which_model == 'lstm':
        m_model = model.LSTM(args, m_embedding)
    elif args.which_model == 'bilstm':
        m_model = model.BILSTM(args, m_embedding)
    elif args.which_model == 'gru':
        m_model = model.GRU(args, m_embedding)
    elif args.which_model == 'bigru':
        m_model = model.BIGRU(args, m_embedding)
    elif args.which_model == 'rnn':
        m_model = model.RNN(args, m_embedding)
    elif args.which_model == 'birnn':
        m_model = model.BIRNN(args, m_embedding)
    elif args.which_model == 'cnn':
        m_model = model.CNN(args, m_embedding)
    elif args.which_model == 'lstmCopyFromA7b23':
        m_model = model.LSTMCopyFromA7b23(args, m_embedding)
    elif args.which_model == 'ConvLSTM':
        m_model = model.ConvLSTM(args, m_embedding)
    elif args.which_model == "mylstm":
        m_model = model.MyLSTM(args, m_embedding)
    elif args.which_model == "mybilstm":
        m_model = model.MyBILSTM(args, m_embedding)
    elif args.which_model == "rnntocnn":
        m_model = model.RNNtoCNN(args, m_embedding)
    elif args.which_model == "lstmattention":
        m_model = model.LSTMAttention(args, m_embedding)
    elif args.which_model == "gruattention":
        m_model = model.GRUAttention(args, m_embedding)

else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        m_model = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()
if args.cuda:
    m_model = m_model.cuda()


# train or predict
assert m_model is not None

if args.predict is not None:
    # label = train.predict(args.predict, m_model, text_field, label_field)
    # print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
    pass
elif args.test:
    try:
        train.eval(test_iter, m_model, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    torch.set_num_threads(1)
    train.train(train_iter, dev_iter, test_iter, m_model, args)
