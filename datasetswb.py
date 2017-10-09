# -*- coding: utf-8 -*-
import re
import os
import random
import torch
from torch.autograd import Variable

random.seed(66)


class Example(object):
    """
    生成每一句语料的
    []list 加上一个 []标签
    这个类仅仅是一个存储结构
    """
    def __init__(self, sequence, label):
        self.sequence = sequence
        self.label = label

    # 专门写一个博客关于各种标记，classmethod。整理笔记
    @classmethod
    def fromlist(cls, sequence, label):
        seq = []
        strs = sequence.split(' ')
        for word in strs:
            seq.append(word)
        return cls(seq, label)


# 利用正则表达式处理每一句话
def cleansequence(string):
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def splitcorpus(path, label_num=2, shuffle=True):
    # 和这句话进行一下对比，看看有啥好处。整理笔记
    # with open(path, encoding='utf-8').readlines() as f:
    examples = []
    f = open(path, encoding='utf-8').readlines()
    if label_num == 2:
        for line in f:
            l = cleansequence(line[:line.find('|')])
            if line[-2] == '0' or line[-2] == '1':
                examples.append(Example.fromlist(l, 'negative'))
            elif line[-2] == '3' or line[-2] == '4':
                examples.append(Example.fromlist(l, 'positive'))
    elif label_num == 5:
        for line in f:
            line = cleansequence(line)
            if line[-2] == '0' or line[-2] == '1':
                examples += Example.fromlist(line[:line.find('|')], 'negative')
            elif line[-2] == '3' or line[-2] == '4':
                examples += Example.fromlist(line[:line.find('|')], 'positive')
    if shuffle:
        random.shuffle(examples)
    return examples


class Vocabulary:
    """
    词表，包含id2word 和 word2id
    但是在建立词表的时候，考虑到可能训练集，开发集，测试集一起建立词表，所以应该输入的一个list
    """
    def __init__(self, id2word, word2id):
        self.id2word = id2word
        self.word2id = word2id

    @classmethod
    def makeVocabularyByText(cls, examplesAll):
        frequence = dict()
        id2word = {}
        word2id = {}
        for examples in examplesAll:
            for e in examples:
                for word in e.sequence:
                    if word in frequence:
                        frequence[word] += 1
                    else:
                        frequence[word] = 1
        # 排序,整理笔记
        # 按照降序排列
        allwords = sorted(frequence.items(), key=lambda t: t[1], reverse=True)
        id2word[0] = "<unknown>"
        word2id["<unknown>"] = 0
        id2word[1] = "<padding>"
        word2id["<padding>"] = 1
        for idx, word in enumerate(allwords):
            """
            0 留给 <unknown>
            1 留给 <padding>
            """
            id2word[idx + 2] = word[0]
            word2id[word[0]] = idx + 2
        """在这里可以加上提取有用的词向量的操作
        但是整体的架构如何设计一下呢？？？？？？
        """
        # save_dir = "D:/AI/embedding&corpus"
        # assert os.path.isdir(save_dir)
        # output = open(os.path.join(save_dir, "glove300d.txt"), "w+", encoding='utf-8')
        # with open("D:/AI/embedding&corpus/glove.840B.300d.txt", encoding="utf-8") as f:
        #     hang = 0
        #     count = 0
        #     find = 0
        #     for line in f:
        #         if hang == 0 or line == "":
        #             hang += 1
        #         else:
        #             line = line.strip()
        #             strs = line.split(' ')
        #             if strs[0] in word2id:
        #                 output.write(line + '\n')
        #                 output.flush()
        #                 find += 1
        #             count += 1
        # output.close()
        # print("find:", find)
        # print("all:", count)
        return cls(id2word, word2id)

    @classmethod
    def makeVocabularyByLable(cls, examplesAll):
        frequence = dict()
        id2word = {}
        word2id = {}
        for examples in examplesAll:
            for e in examples:
                if e.label in frequence:
                    frequence[e.label] += 1
                else:
                    frequence[e.label] = 1
        # 排序,整理笔记
        # 按照降序排列
        allwords = sorted(frequence.items(), key=lambda t: t[1], reverse=True)
        for idx, word in enumerate(allwords):
            """
            0 留给 <unknown>
            1 留给 <padding>
            """
            id2word[idx] = word[0]
            word2id[word[0]] = idx
        return cls(id2word, word2id)


class Batch:
    """
    仅仅用来存储text,label
    还要对padding进行处理
    """
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.batch_size = len(text)

    @classmethod
    def makeBatch(cls, item):
        """
        是不是也可以去平均值得长度。padding太多感觉会有问题
        这个问题是不是世纪难题
        填充1<padding>
        """
        max_len = 0
        for i in item:
            if len(i[0]) > max_len:
                max_len = len(i[0])
        text = []
        label = []
        for i in item:
            seq = []
            for idx in range(max_len):
                if idx < len(i[0]):
                    seq.append(i[0][idx])
                else:
                    seq.append(1)
            text.append(seq)
            label.append(i[1][0])
        # 转变成Variable
        text = Variable(torch.LongTensor(text))
        label = Variable(torch.LongTensor(label))
        return cls(text, label)


class MyIterator:
    """
    迭代器，返回一个list，里面存储着Variable，每个Variable的size是
    batch * （batch个句子中最长的那个句子的大小）
    转换成id表示
    """
    def __init__(self, batch_size, examples, vocabulary_text, vocabulary_label):
        # 按照batch_size所有的
        self.iterators = []
        # 一个batch
        item = []
        # 计数，每batch_size的倍数生成一次Batch
        count = 0
        for example in examples:
            text = []
            label = []
            for word in example.sequence:
                if word in vocabulary_text.word2id:
                    text.append(vocabulary_text.word2id[word])
                else:
                    # dev & test 的数据可能不在vocabulary中，填充0<unknown>
                    text.append(0)
            label.append(vocabulary_label.word2id[example.label])
            item.append((text, label))
            count += 1
            if count % batch_size == 0 or count == len(examples):
                self.iterators.append(Batch.makeBatch(item))
                item = []


class MyDatasets:
    """
    只要最后返回一个一个根据batch的迭代器就好了
    """
    def __init__(self, args, path, label_num, shuffle=True, examples=None):
        # 生成每一个标准形式的句子
        if examples is None:
            self.examples = splitcorpus(path, label_num, shuffle)
        else:
            self.examples = examples
        # 建立词表
        # 也没有必要非要在这里建立词表，也可以单写
        vocabulary_text = Vocabulary.makeVocabularyByText([self.examples])
        # output = open("D:/vocab.txt", "w+", encoding='utf-8')
        # for k in vocabulary_text.word2id:
        #     output.write(k + '\n')
        #     output.flush()
        # output.close()

        vocabulary_label = Vocabulary.makeVocabularyByLable([self.examples])
        """
        感觉这里必然会有一些问题存在
        """
        # 生成迭代器，batch
        batch_size = args
        iterator = MyIterator(batch_size, self.examples, vocabulary_text, vocabulary_label)
        print("sdfs")


if __name__ == "__main__":
    print('-----------')
    args = 64
    path = "./data/raw.clean.train"
    data = MyDatasets(args, path, 2)
