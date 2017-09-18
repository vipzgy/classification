# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args, m_embedding):
        super(CNN, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = [int(i) for i in args.kernel_sizes if not i == ',']

        self.embed = nn.Embedding(V, D)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]

        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        x = self.embed(x)

        x = torch.unsqueeze(x, 1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        logit = self.fc1(x)
        return logit
