# -*- coding: utf-8 -*-
import re
import os
import random
from torchtext import data

random.seed(66)


class MR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, label5=False, **kwargs):
        def clean_str(string):
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

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []

            if not label5:
                with open(os.path.join('./data', path), encoding="utf-8") as f:
                    for line in f.readlines():
                        if line[-2] == '0' or line[-2] == '1':
                            examples += [
                                data.Example.fromlist([line[:line.find('|')], 'negative'], fields)]
                        elif line[-2] == '3' or line[-2] == '4':
                            examples += [
                                data.Example.fromlist([line[:line.find('|')], 'positive'], fields)]
            else:
                with open(os.path.join('./data', path), encoding="utf-8") as f:
                    for line in f.readlines():
                        if line[-2] == '0':
                            examples += [data.Example.fromlist([line[:line.find('|')], 'snegative'], fields)]
                        elif line[-2] == '1':
                            examples += [data.Example.fromlist([line[:line.find('|')], 'wnegative'], fields)]
                        elif line[-2] == '2':
                            examples += [data.Example.fromlist([line[:line.find('|')], 'neutral'], fields)]
                        elif line[-2] == '3':
                            examples += [data.Example.fromlist([line[:line.find('|')], 'wpositive'], fields)]
                        elif line[-2] == '4':
                            examples += [data.Example.fromlist([line[:line.find('|')], 'spositive'], fields)]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True, label5=False, **kwargs):
        example1 = cls(text_field, label_field, path='raw.clean.train', label5=label5, **kwargs).examples
        example2 = cls(text_field, label_field, path='raw.clean.dev', label5=label5, **kwargs).examples
        example3 = cls(text_field, label_field, path='raw.clean.test', label5=label5, **kwargs).examples
        if shuffle:
            random.shuffle(example1)
            random.shuffle(example2)
            random.shuffle(example3)

        return (cls(text_field, label_field, examples=example1),
                cls(text_field, label_field, examples=example2),
                cls(text_field, label_field, examples=example3))
