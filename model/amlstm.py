# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyAttention(nn.Module):
    def __init__(self, args, m_embedding):
        super(MyAttention, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True)

        self.myw = Variable(torch.randn(args.hidden_size, 1))

        self.linearOut = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x):
        x = self.embed(x)

        x, lstm_h = self.lstm(x)

        x = torch.transpose(x, 0, 1)

        for idx in range(x.size(0)):
            tem = torch.mm(x[idx], self.myw)
            tem = torch.exp(F.tanh(tem))
            if idx == 0:
                probability = tem
            else:
                probability = torch.cat([probability, tem], 1)
        max = []
        for idx in range(probability.size(0)):
            max_value = -1
            max_id = -1
            for idj in range(probability.size(1)):
                if probability.data[idx][idj] > max_value:
                    max_id = idj
                    max_value = probability.data[idx][idj]
            max.append(max_id)

        x = torch.transpose(x, 0, 1)

        for idx in range(x.size(0)):
            if idx == 0:
                output = torch.unsqueeze(x[idx][max[idx]], 0)
            else:
                output = torch.cat([output, torch.unsqueeze(x[idx][max[idx]], 0)], 0)
        """
        留个坑，想知道自己选的到底是一句话中的哪一个单词
        """

        x = self.linearOut(output)
        return x