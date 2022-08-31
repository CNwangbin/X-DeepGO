import torch as th
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DGELModel(nn.Module):

    def __init__(self, normal_forms, seq_dim, nb_gos, nb_zero_gos, nb_rels, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__()
        self.normal_forms = normal_forms
        self.nb_gos = nb_gos
        self.nb_zero_gos = nb_zero_gos
        input_length = seq_dim
        net = []
        net.append(MLPBlock(input_length, hidden_dim))
        net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        self.net = nn.Sequential(*net)

        # ELEmbeddings
        self.embed_dim = embed_dim
        self.hasFuncIndex = th.LongTensor([nb_rels]).cuda()
        self.go_embed = nn.Embedding(nb_gos + nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).cuda()
        self.margin = margin

        
    def forward(self, embeddings, labels):
        x = self.net(embeddings)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_gos),
                            labels.view(-1, self.nb_gos))
            loss += self.el_loss()
            outputs = (loss, ) + outputs
    
        return outputs

    def predict_zero(self, features, data):
        x = self.net(features)
        go_embed = self.go_embed(data)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(data).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits


    def el_loss(self):
        nf1, nf2, nf3, nf4 = self.normal_forms
        nf1_loss = self.nf1_loss(nf1)
        nf2_loss = self.nf2_loss(nf2)
        nf3_loss = self.nf3_loss(nf3)
        nf4_loss = self.nf4_loss(nf4)
        # print()
        # print(nf1_loss.detach().item(),
        #       nf2_loss.detach().item(),
        #       nf3_loss.detach().item(),
        #       nf4_loss.detach().item())
        return nf1_loss + nf3_loss + nf4_loss + nf2_loss

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
    def nf1_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))
        
        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin)
                    + th.relu(dst2 - rc - self.margin)
                    + th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        # rS = self.rel_space(data[:, 0])
        # rS = rS.reshape(-1, self.embed_dim, self.embed_dim)
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        # c = th.matmul(c, rS).reshape(n, -1)
        # d = th.matmul(d, rS).reshape(n, -1)
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        
        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss


    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))
        return loss
    
    