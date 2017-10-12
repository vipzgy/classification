# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rnn=None, bias=True):
        super(MyLSTMCell, self).__init__()
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


class MyLSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(MyLSTM, self).__init__()

        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.lstm = MyLSTMCell(args.input_size, args.hidden_size, args.dropout_rnn)

        self.linearout = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)

        x = torch.transpose(x, 0, 1)

        h_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        c_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))

        for idx in range(x.size(0)):
            input_cell = x[idx]
            h_i, c_i = self.lstm(input_cell, h_i, c_i)
            if idx == 0:
                output = torch.unsqueeze(h_i, 0)
            else:
                output = torch.cat([output, torch.unsqueeze(h_i, 0)], 0)

        output = torch.transpose(output, 0, 1)
        output = F.tanh(torch.transpose(output, 1, 2))
        output = F.max_pool1d(output, output.size(2)).squeeze(2)

        output = self.linearout(output)
        return output


class MyBILSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(MyBILSTM, self).__init__()

        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        # 这是一个双向lstm
        self.lstm_cell_forward = MyLSTMCell(args.input_size, args.hidden_size, args.dropout_rnn)
        self.lstm_cell_backward = MyLSTMCell(args.input_size, args.hidden_size, args.dropout_rnn)

        self.linearout = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)

        x = torch.transpose(x, 0, 1)

        h_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        c_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        if self.args.cuda:
            h_i.cuda()
            c_i.cuda()

        for idx in range(x.size(0)):
            input_cell = x[idx]
            h_i, c_i = self.lstm_cell_forward(input_cell, h_i, c_i)
            if idx == 0:
                output_1 = torch.unsqueeze(h_i, 0)
            else:
                output_1 = torch.cat([output_1, torch.unsqueeze(h_i, 0)], 0)

        h_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        c_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        if self.args.cuda:
            h_i.cuda()
            c_i.cuda()

        idx = x.size(0) - 1
        while idx >= 0:
            input_cell = x[idx]
            h_i, c_i = self.lstm_cell_backward(input_cell, h_i, c_i)

            if idx == x.size(0) - 1:
                output_2 = torch.unsqueeze(h_i, 0)
            else:
                output_2 = torch.cat([output_2, torch.unsqueeze(h_i, 0)], 0)

            idx -= 1

        # 直接concat
        # output = torch.transpose(torch.cat([output_1, output_2], 2), 0, 1)
        # output = F.tanh(torch.transpose(output, 1, 2))
        # output = F.max_pool1d(output, output.size(2)).squeeze(2)

        # 对应位置上的
        m_len = output_1.size(0)
        for idx in range(output_1.size(0)):
            # mul
            t = torch.mul(output_1[idx], output_2[m_len - idx - 1])
            # average
            # t = 0.5 * (output_1[idx] + output_2[m_len - idx - 1])
            if idx == 0:
                output = torch.unsqueeze(t, 0)
            else:
                output = torch.cat([output, torch.unsqueeze(t, 0)], 0)
        output = torch.transpose(output, 0, 1)
        output = F.tanh(torch.transpose(output, 1, 2))
        output = F.max_pool1d(output, output.size(2)).squeeze(2)

        output = self.linearout(output)
        return output







