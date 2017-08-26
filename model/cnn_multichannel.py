# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_MC(nn.Module):
    def __init__(self, args, m_embedding):
        super(CNN_MC, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 2
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed1 = nn.Embedding(V, D)
        self.embed2 = nn.Embedding(V, D)
        self.embed1.weight.data.copy_(m_embedding)
        self.embed2.weight.data.copy_(m_embedding)
        # self.embed2.weight.requires_grad = False
        # 更新了之后是
        # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x1 = self.embed1(x)
        x2 = self.embed2(x)

        if self.args.static:
            x2 = Variable(x2.data)
        if self.args.cuda:
            self.convs1 = [model.cuda() for model in self.convs1]

        x = torch.stack([x1, x2], 1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
