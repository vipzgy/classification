# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MyLSTMCell, self).__init__()

        # 先不要激活函数,激活函数没有参数，可以放在forward里面用F
        # 这里是直接带入batch计算的
        if bias is False:
            self.linearf = nn.Linear(input_size + hidden_size, hidden_size)
            self.lineari = nn.Linear(input_size + hidden_size, hidden_size)
            self.linearo = nn.Linear(input_size + hidden_size, hidden_size)
            self.linearc = nn.Linear(input_size + hidden_size, hidden_size)
        else:
            self.linearf = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.lineari = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.linearo = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            self.linearc = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

    def forward(self, xt, ht_pro, ct_pro):
        ft = F.sigmoid(self.linearf(torch.cat([xt, ht_pro], 1)))
        it = F.sigmoid(self.lineari(torch.cat([xt, ht_pro], 1)))
        c_t = F.tanh(self.linearc(torch.cat([xt, ht_pro], 1)))
        ot = F.sigmoid(self.linearo(torch.cat([xt, ht_pro], 1)))

        ct = torch.mul(ft, c_t) + torch.mul(it, c_t)
        ht = torch.mul(ot, F.tanh(ct))

        return ht, ct


class MyLSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(MyLSTM, self).__init__()

        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        # 这是一个双向lstm
        self.lstm_cell_forward = MyLSTMCell(args.input_size, args.hidden_size)
        self.lstm_cell_backward = MyLSTMCell(args.input_size, args.hidden_size)

        self.linearout = nn.Linear(args.hidden_size * 2, args.class_num)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)

        x = torch.transpose(x, 0, 1)

        h_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        c_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))

        for idx in range(x.size(0)):
            input_cell = x[idx]
            h_i, c_i = self.lstm_cell_forward(input_cell, h_i, c_i)
            if idx == 0:
                output_1 = torch.unsqueeze(h_i, 0)
            else:
                output_1 = torch.cat([output_1, torch.unsqueeze(h_i, 0)], 0)

        h_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        c_i = Variable(torch.zeros(x.size(1), self.args.hidden_size))
        idx = x.size(0) - 1
        while idx >= 0:
            input_cell = x[idx]
            h_i, c_i = self.lstm_cell_backward(input_cell, h_i, c_i)

            if idx == x.size(0) - 1:
                output_2 = torch.unsqueeze(h_i, 0)
            else:
                output_2 = torch.cat([output_2, torch.unsqueeze(h_i, 0)], 0)

            idx -= 1

        output = torch.transpose(torch.cat([output_1, output_2], 2), 0, 1)
        output = F.tanh(torch.transpose(output, 1, 2))
        output = F.max_pool1d(output, output.size(2)).squeeze(2)

        output = self.linearout(output)
        return output







