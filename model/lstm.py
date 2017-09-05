# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(66)


class LSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(LSTM, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True, bidirectional=True)
        # 使用Xavier初始化，也就这一个是有weight
        # nn.init.xavier_normal(self.lstm.all_weights[0][0], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[0][1], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[1][0], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[1][1], 1)

        self.linearOut = nn.Linear(args.hidden_size * 2, args.class_num)
        # nn.init.xavier_normal(self.linearOut.weight, 1)

    def forward(self, x):
        # hidden = Variable(torch.zeros(2, x.size(0), self.args.hidden_size))
        hidden = (Variable(torch.zeros(2, x.size(0), self.args.hidden_size)),
                  Variable(torch.zeros(2, x.size(0), self.args.hidden_size)))
        x = self.embed(x)
        x = self.dropout(x)

        x, lstm_h = self.lstm(x, hidden)

        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linearOut(x)
        return x


class BILSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(BILSTM, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True, bidirectional=True)
        # 使用Xavier初始化，也就这一个是有weight
        # nn.init.xavier_normal(self.lstm.all_weights[0][0], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[0][1], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[1][0], 1)
        # nn.init.xavier_normal(self.lstm.all_weights[1][1], 1)

        self.linearOut = nn.Linear(args.hidden_size * 2, args.class_num)
        # nn.init.xavier_normal(self.linearOut.weight, 1)

    def forward(self, x):
        # hidden = Variable(torch.zeros(2, x.size(0), self.args.hidden_size))
        hidden = (Variable(torch.zeros(2, x.size(0), self.args.hidden_size)),
                  Variable(torch.zeros(2, x.size(0), self.args.hidden_size)))
        x = self.embed(x)
        x = self.dropout(x)

        x, lstm_h = self.lstm(x, hidden)

        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.linearOut(x)
        return x