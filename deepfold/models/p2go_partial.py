import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfold.models.gnn_model import GCN
from deepfold.models.esm_model import MLPLayer3D, MLPLayer
from torch.nn import BCEWithLogitsLoss

import math
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from deepfold.models.esm_model import MLPLayer



# attention  module
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项 Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作."""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """缩放点积注意力."""
    def __init__(self, dropout, lambd=1,**kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.lambd = lambd
        self.dropout = nn.Dropout(dropout)
    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.lambd * d)
        # scores /= self.tao
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values),self.attention_weights

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状."""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力."""
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 lambd=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout,lambd=lambd)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None, output_attentions=True):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output,weight = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        weight_concat = transpose_output(weight, self.num_heads)
        outputs = (output_concat, weight_concat) if output_attentions else (output_concat,)
            
 
        return outputs

class P2GO(nn.Module):
    def __init__(self,
                 terms_embedding,adj,
                 model_dir: str = 'esm1b_t33_650M_UR50S',
                 aa_dim=1280,
                 latent_dim=256,
                 dropout_rate=0.1,
                 n_head=2,lambd=10):
        super().__init__()
        self.terms_embedding = nn.Parameter(terms_embedding,requires_grad=True)
        self.terms_dim = terms_embedding.shape[1]
        self.latent_dim = latent_dim
        self.adj = adj
        
        # backbone
        backbone, _ = esm.pretrained.load_model_and_alphabet(
            model_dir)
        unfreeze_layers = [32] # total 0-32 layer
        self.backbone = self.unfreeze(backbone,unfreeze_layers)
        self.nb_classes = adj.shape[0]
        # AA_emebdding transform
        self.aa_transform = MLPLayer3D(aa_dim, latent_dim)
        # go transform
        self.go_transfrom = MLPLayer(self.terms_dim,self.latent_dim)
        # label-wise attention
        self.attention = MultiHeadAttention(latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            num_heads=n_head,
                                            dropout=dropout_rate,
                                            lambd=lambd) # lambd is a hyper parameters
        # gnn module
        self.gcn = GCN(latent_dim, latent_dim)
        # output layer
        self.go_transform_post = MLPLayer(int(2 * latent_dim), latent_dim)

    def unfreeze(self, backbone, unfreeze_layers:list):
        for name ,param in backbone.named_parameters():
            param.requires_grad = False
        if unfreeze_layers is not None:
            if 'lm_head' in name:
                param.requires_grad = True
            for idx in unfreeze_layers:
                for _, p in backbone.layers[idx].named_parameters():
                    p.requires_grad = True
        return backbone

    def forward(self, input_ids, lengths, labels, output_attention_weights=True):
        # backbone
        x = self.backbone(input_ids, repr_layers=[33])['representations'][33]
        # x = x[:, 1:]
        # x [B,L,C]
        # AA_embedding transform
        x = self.aa_transform(x)
        # mean_list = []
        # for i, length in enumerate(list(lengths)):
        #     length = int(length)
        #     mean = x[i, 1:length+1].mean(axis=0, keepdim=True)
        #     mean_list.append(mean)
        # mean_batch = torch.cat(mean_list)
        # go embedder
        # go_embedding [nb_classes,latent_dim]
        go_embedding = self.go_transfrom(self.terms_embedding)
        # mean preds
        # mean_logits = torch.mm(mean_batch,go_embedding.T)
        # label-wise attention
        y_embedding = go_embedding.repeat((x.shape[0], 1, 1))
        label_attn_embedding, weights = self.attention(y_embedding, x, x, lengths)
        # go embedding
        go_out = self.gcn(go_embedding, self.adj)
        # output layer
        go_out = torch.cat((go_embedding, go_out), dim=1)
        go_out = self.go_transform_post(go_out)
        go_out = go_out.repeat((x.shape[0], 1, 1))
        logits = torch.sum(go_out * label_attn_embedding, dim=-1)
        # logits += mean_logits
        outputs = logits
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))
            outputs = (logits,loss)
        if output_attention_weights:
            outputs = (loss, logits, weights)

        return outputs


if __name__ == '__main__':
    pass
    # namespace = 'mfo'
    # adj, multi_hot_vector, label_map, label_map_ivs,_ = build_graph(
    #     data_path=data_path, namespace=namespace)
    # one_hot, anc_label,sub_ont_label = generator(label_map,go_file)
    
    
    # one_hot = one_hot.cuda()
    # sub_ont_label = sub_ont_label.cuda()
    # anc_label = anc_label.cuda()
    # anc2vec = Anc2vec(one_hot=one_hot,sub_ont_label=sub_ont_label,anc_label=anc_label)
    # anc2vec = anc2vec.cuda()
    # optimizer = optim.AdamW(params=anc2vec.parameters(),lr=1e-3)
    # anc2vec.train()
    # for i in range(10000):
    #     optimizer.zero_grad()
    #     label_embedding, loss = anc2vec()
    #     loss.backward()
    #     optimizer.step()
    #     print(loss)
    
    from deepfold.utils.make_graph import build_graph
    from deepfold.data.esm_dataset import EsmDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import os
    import pandas as pd
    import numpy as np

    data_path = '../../data/cafa3/'
    go_file = '../../data/cafa3/go_cafa3.obo'

    # CAFA3 esm dataset
    mfo_dataset = EsmDataset(data_path='../../data/cafa3/mfo/',
                             file_name='mfo_train_data.pkl',
                             terms_name='mfo_terms.pkl')
    mfo_loader = DataLoader(mfo_dataset,
                            batch_size=8,
                            collate_fn=mfo_dataset.collate_fn)
    for index, batch in enumerate(mfo_loader):
        for key, val in batch.items():
            print(key, val.shape)
        break
    batch = next(iter(mfo_loader))
    batch = {key: val.cuda() for key, val in batch.items()}

    # generate anc2vec train data
    namespace = 'mfo'
    adj, multi_hot_vector, label_map, label_map_ivs,_ = build_graph(
        data_path=data_path, namespace=namespace)
    terms_all = pd.read_pickle(os.path.join(data_path,'all_terms_partial_order_embeddings.pkl'))
    terms = pd.read_pickle(os.path.join(data_path,namespace,namespace + '_terms.pkl'))
    terms_embedding = terms.merge(terms_all)

    embeddings = np.concatenate([np.array(embedding,ndmin=2) for embedding in terms_embedding.embeddings.values])
    terms_embedding = torch.Tensor(embeddings)
    terms_embedding = terms_embedding.cuda()
    adj = adj.cuda()
    model = P2GO(terms_embedding=terms_embedding,adj=adj,latent_dim=256,n_head=2,lambd=10)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=1e-4)
    model = model.cuda()
    model.train()
    optimizer.zero_grad()


    outputs = model(**batch)
    loss = outputs[0]
    loss.backward()
    optimizer.step()
    print(f'logits:{outputs[1].shape}')
    print(f'loss:{loss}')
    print(f'attention weights:{outputs[2].shape}')

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,param.size())
    # for index, batch in enumerate(mfo_loader):
    #     batch = {key: val.cuda() for key, val in batch.items()}
    #     optimizer.zero_grad()
    #     outputs = model(**batch)
    #     loss = outputs[1]
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())


