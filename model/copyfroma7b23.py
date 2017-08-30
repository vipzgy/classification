# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(66)


class LSTMCopyFromA7b23(nn.Module):
    def __init__(self, args, m_embedding):
        super(LSTMCopyFromA7b23, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        self.lstm = nn.LSTM(args.input_size, args.hidden_size, batch_first=True)
        self.linearOut = nn.Linear(args.hidden_size, args.class_num)

        # self.dropout = nn.Dropout(args.dropout_embed)
        #
        # self.lstm = nn.LSTM(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True, bidirectional=True)
        #
        # self.linearOut = nn.Linear(args.hidden_size * 2, args.class_num)

    def forward(self, x):
        hidden = (Variable(torch.zeros(1, x.size(0), self.args.hidden_size)),
                  Variable(torch.zeros(1, x.size(0), self.args.hidden_size)))
        x = self.embed(x)
        lstm_out, lstm_h = self.lstm(x, hidden)
        x = torch.transpose(lstm_out, 0, 1)
        x = x[-1]
        x = self.linearOut(x)
        x = F.log_softmax(x)
        return x