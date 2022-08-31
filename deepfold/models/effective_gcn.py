import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Embedder(nn.Module):
    def __init__(self, go_size, hidden_dimension):
        super().__init__()
        self.embed = nn.Linear(go_size,hidden_dimension)

    def forward(self, x):
        node_feature = self.embed(x)
        node_feature = F.normalize(node_feature)
        return node_feature


class FC(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        seq_feature = self.Linear(x)

        return seq_feature

class GraphConvolution(nn.Module):
    def __init__(self, nfeat, nhid, bias=True):
        super(GraphConvolution, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        if bias:
            self.bias = Parameter(torch.FloatTensor(nhid))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = torch.mm(input, self.weight)
        output = torch.spmm(adj, x)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x

class EffectiveGCNModel(nn.Module):
    def __init__(self,
                 nodes: torch.Tensor,
                 adjmat: torch.Tensor,
                 seq_dim: int = 1024,
                 node_feats: int = 512,
                 hidden_dim: int = 512):
        super().__init__()
        assert nodes.shape[0] == adjmat.shape[0]
        self.nodesMat = nodes
        self.adjMat = adjmat
        self.num_nodes = nodes.shape[0]
        self.seq_mlp = FC(seq_dim, hidden_dim)
        self.graph_embedder = Embedder(self.num_nodes, node_feats)
        self.gcn = GCN(node_feats, hidden_dim)
        self.num_labels = self.num_nodes

    def forward(self, embeddings, labels):
        seq_out = self.seq_mlp(embeddings)
        node_embd = self.graph_embedder(self.nodesMat)
        graph_out = self.gcn(node_embd, self.adjMat)
        graph_out = graph_out.transpose(-2, -1)
        logits = torch.matmul(seq_out, graph_out)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs