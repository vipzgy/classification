# -*- coding: utf-8 -*-
import os
import pickle
import random
import argparse
import datetime

import numpy
import torch
import torchtext.data as data

import model
import train
import mydatasets

random.seed(66)
torch.manual_seed(66)

parser = argparse.ArgumentParser(description='classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=8)
parser.add_argument('-batch-size', type=int, default=16)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=100)
parser.add_argument('-save-interval', type=int, default=100)
parser.add_argument('-save-dir', type=str, default='snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=True)
# model
parser.add_argument('-dropout-embed', type=float, default=0.5)
parser.add_argument('-dropout-rnn', type=float, default=0.6)

parser.add_argument('-use-embedding', action='store_true', default=True)
parser.add_argument('-max-norm', type=float, default=None)
parser.add_argument('-embed-dim', type=int, default=300)

parser.add_argument('-input-size', type=int, default=300)
parser.add_argument('-hidden-size', type=int, default=200)

parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False)

parser.add_argument('-which-model', type=str, default='gru')
# device
parser.add_argument('-device', type=int, default=-1)
parser.add_argument('-no-cuda', action='store_true', default=True)
# option
parser.add_argument('-snapshot', type=str, default=None)
parser.add_argument('-predict', type=str, default=None)
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-label5', action='store_true', default=False)
args = parser.parse_args()


# load dataset
def mr(text_field, label_field, label5, **kargs):
    train_data, dev_data, test_data = mydatasets.MR.splits(text_field, label_field, label5=label5)
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        **kargs)
    return train_iter, dev_iter, test_iter


# create embedding
def getEmbedding(plk_path, embed_path, id2word, name):
    if os.path.exists(plk_path):
        plk_f = open(plk_path, 'rb+')
        m_embed = pickle.load(plk_f)
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)
        plk_f.close()
    else:
        assert os.path.exists(embed_path)
        embed_f = open(embed_path, encoding="utf-8")
        m_dict = {}
        for idx, line in enumerate(embed_f.readlines()):
            if not (idx == 0 or line == ''):
                strs = line.split(' ')
                m_dict[strs[0]] = [float(i) for idx2, i in enumerate(strs) if not idx2 == 0]
        embed_f.close()

        m_embed = [m_dict['unknown']]
        notfound = 0
        for idx, word in enumerate(id2word):
            if not idx == 0:
                if word in m_dict:
                    m_embed.append(m_dict[word])
                else:
                    notfound += 1
                    m_embed.append([random.uniform(-0.25, 0.25) for i in range(args.embed_dim)])
        print('notfound:', notfound)
        print('ratio:', notfound / (len(id2word) - 1))
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)

        f = open(os.path.join('./data', name), 'wb+')
        # pickle.dump(id2word, f)
        pickle.dump(m_embed, f)
        f.close()
    return m_embedding


# load data
print("\nLoading data...")
# ????
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = mr(text_field, label_field,
                                     device=args.device,
                                     repeat=False,
                                     shuffle=args.shuffle,
                                     label5=args.label5)


# load embedding
m_embedding = None
if args.use_embedding:
    id2word = text_field.vocab.itos
    m_embedding = getEmbedding('./data/conj300d.pkl',
                               './data/glove.sentiment.conj.pretrained.txt',
                               id2word,
                               'conj300d.pkl')
    # m_embedding = getEmbedding('./data/conj300d.pkl',
    #                            './data/glove.sentiment.conj.pretrained.txt',
    #                            id2word,
    #                            'conj300d.pkl')


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
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
    elif args.which_model == 'gru':
        m_model = model.GRU(args, m_embedding)
    elif args.which_model == 'rnn':
        m_model = model.RNN(args, m_embedding)
    elif args.which_model == 'cnn':
        m_model = model.CNN(args, m_embedding)
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
    label = train.predict(args.predict, m_model, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, m_model, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    torch.set_num_threads(3)
    train.train(train_iter, dev_iter, m_model, args)

    # 直接测试所有的模型，选出最好的
    m_max = -99999
    whichmax = ''
    dirlist = os.listdir(args.save_dir)
    f = open(os.path.join(args.save_dir, 'testresult'), "w+", encoding='utf-8')
    for attr, value in sorted(args.__dict__.items()):
        f.write("\t{}={} \n".format(attr.upper(), value))
        f.flush()
    f.write('----------------------------------------------------')
    f.flush()
    for name in dirlist:
        t_model = torch.load(os.path.join(args.save_dir, name))
        m_str, accuracy = train.test(test_iter, t_model, args)
        f.write(m_str + '-------' + name + '\n')
        f.flush()
        if accuracy > m_max:
            m_max = accuracy
            whichmax = name
    f.write('max is {} using {}'.format(m_max, whichmax))
    f.flush()
    f.close()
