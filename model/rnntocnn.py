import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNNtoCNN(nn.Module):
    def __init__(self, args, m_embedding):
        super(RNNtoCNN, self).__init__()

        self.args = args

        V = args.embed_num
        D = args.embed_dim
        max_norm = args.max_norm
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = [int(i) for i in args.kernel_sizes if not i == ',']

        self.embed = nn.Embedding(V, D, max_norm=max_norm)
        if args.use_embedding:
            self.embed.weight.data.copy_(m_embedding)

        # 就是感觉不是很需要长距离依赖
        self.rnn = nn.RNN(args.input_size, args.hidden_size, dropout=args.dropout_rnn, batch_first=True)

        self.conv1s = [nn.Conv2d(Ci, Co, (K, args.hidden_size)) for K in Ks]

        self.linear1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        result = self.embed(x)
        result, _ = self.rnn(result)
        result = torch.unsqueeze(result, 1)

        result = [F.tanh(conv(result)).squeeze(3) for conv in self.conv1s]
        result = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in result]
        result = torch.cat(result, 1)

        result = self.linear1(result)
        return result