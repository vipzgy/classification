# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import numpy
import torch
import torchtext.data as data

import model
import train
import mydatasets

import pickle

torch.manual_seed(66)

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.0015, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=200, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-input-size', type=int, default=100, help='The number of expected features in the input x')
parser.add_argument('-hidden-size', type=int, default=60, help='The number of features in the hidden state h')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=True, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-label5', action='store_true',)
args = parser.parse_args()


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data, test_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data)
    label_field.build_vocab(train_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train_data, dev_data, test_data),
                                batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
                                **kargs)
    return train_iter, dev_iter, test_iter

# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = mr(text_field, label_field, device=-1, repeat=False, shuffle=False)

'''
做法是根据生成的itos来生成embedding，
然后保存一个pickle，如果有pickle就不执行这个了
'''
id2word = text_field.vocab.itos
f = open("./data/glove.6B.100d.txt", encoding="utf-8").readlines()
m_dict = {}
for idx, line in enumerate(f):
    if not (idx == 0 or line == ''):
        strs = line.split(' ')
        m_dict[strs[0]] = [float(i) for idx2, i in enumerate(strs) if not idx2 == 0]

m_embed = [m_dict['unknown']]
notfound = 0
for idx, word in enumerate(id2word):
    if not idx == 0:
        if word in m_dict:
            m_embed.append(m_dict[word])
        else:
            notfound += 1
            m_embed.append([0 for i in range(args.embed_dim)])
print('notfound:', notfound)
print('ratio:', notfound/(len(id2word)-1))
m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)

# f = open('./data/pickle.txt', 'rb+')
# m_embed = pickle.load(f)
# m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.snapshot is None:
    lstm = model.LSTM_Text(args, m_embedding)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        lstm = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist."); exit()
if args.cuda:
    lstm = lstm.cuda()

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, lstm, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, lstm, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    torch.set_num_threads(5)
    train.train(train_iter, dev_iter, lstm, args)


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
        model = torch.load(os.path.join(args.save_dir, name))
        str, accuracy = train.test(test_iter, model, args)
        f.write(str + '-------' + name + '\n')
        f.flush()
        if accuracy > m_max:
            m_max = accuracy
            whichmax = name
    f.write('max is {} using {}'.format(m_max, whichmax))
    f.flush()
    f.close()

