# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

torch.manual_seed(66)


class RNN(nn.Module):
    def __init__(self, args, m_embedding):
        super(RNN, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout)

        self.rnn = nn.RNN(args.input_size, args.hidden_size, dropout=args.dropout, batch_first=True, bidirectional=True)

        self.linearOut = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x, hidden):
        x = self.embed(x)
        x = self.dropout(x)

        x, lstm_h = self.rnn(x, hidden)

        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linearOut(x)
        return x, lstm_h

    def init_hidden(self, batch):
        return Variable(torch.zeros(1, batch, self.args.hidden_size))
