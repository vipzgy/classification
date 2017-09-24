# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRUAttention(nn.Module):
    def __init__(self, args, m_embedding):
        super(GRUAttention, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        self.dropout = nn.Dropout(args.dropout_embed)

        self.gru = nn.GRU(args.input_size, args.hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          dropout=args.dropout_rnn)
        nn.init.kaiming_uniform(self.gru.all_weights[0][0])
        nn.init.kaiming_uniform(self.gru.all_weights[0][1])
        nn.init.kaiming_uniform(self.gru.all_weights[1][0])
        nn.init.kaiming_uniform(self.gru.all_weights[1][1])

        self.myw = Variable(torch.randn(args.hidden_size * 2, 1))

        # self.mybias = Variable(torch.randn(args.hidden_size * 2, 1))

        self.linear1 = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.linear2 = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x):
        x = self.embed(x)

        x = self.dropout(x)

        x, _ = self.gru(x)

        x = torch.transpose(x, 0, 1)

        for idx in range(x.size(0)):
            tem = torch.mm(x[idx], self.myw)
            tem = torch.exp(F.tanh(tem))
            if idx == 0:
                probability = tem
            else:
                probability = torch.cat([probability, tem], 1)

        x = torch.transpose(x, 0, 1)
        all_score = []
        for idx in range(probability.size(0)):
            score = 0
            for idj in range(probability.size(1)):
                score += probability.data[idx][idj]
            all_score.append(score)

        for idx in range(probability.size(0)):
            if idx == 0:
                output = torch.mm(torch.unsqueeze(probability[idx], 0)/all_score[idx], x[idx])
            else:
                output = torch.cat([output, torch.mm(torch.unsqueeze(probability[idx], 0)/all_score[idx], x[idx])], 0)

        x = self.linear1(output)
        x = self.linear2(x)
        return x