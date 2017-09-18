# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyAttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rnn=None, bias=True):
        super(MyAttentionCell, self).__init__()
        self.dropout = dropout_rnn

        if bias is False:
            self.linearf = nn.Linear(input_size + hidden_size, hidden_size, )
            self.lineari = nn.Linear(input_size + hidden_size, hidden_size)
            self.linearo = nn.Linear(input_size + hidden_size, hidden_size)
            self.linearc = nn.Linear(input_size + hidden_size, hidden_size)
        else:
            self.linearf = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.lineari = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.linearo = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.linearc = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout_rnn)

    def forward(self, xt, ht_pro, ct_pro):
        ft = F.sigmoid(self.linearf(torch.cat([xt, ht_pro], 1)))
        it = F.sigmoid(self.lineari(torch.cat([xt, ht_pro], 1)))
        c_t = F.tanh(self.linearc(torch.cat([xt, ht_pro], 1)))
        ot = F.sigmoid(self.linearo(torch.cat([xt, ht_pro], 1)))

        ct = torch.mul(ft, c_t) + torch.mul(it, c_t)
        ht = torch.mul(ot, F.tanh(ct))

        if self.dropout is not None:
            ht = self.dropout(ht)
            ct = self.dropout(ct)
        return ht, ct


class MyAttention(nn.Module):
    def __init__(self, args, m_embedding):
        super(MyAttention, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True)

        self.linearOut = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x):
        # hidden = Variable(torch.zeros(2, x.size(0), self.args.hidden_size))
        hidden = (Variable(torch.zeros(1, x.size(0), self.args.hidden_size)),
                  Variable(torch.zeros(1, x.size(0), self.args.hidden_size)))
        x = self.embed(x)
        # x = self.dropout(x)

        x, lstm_h = self.lstm(x, hidden)

        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linearOut(x)
        return x