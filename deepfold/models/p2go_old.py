import math
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfold.models.gnn_model import GCN
from deepfold.utils.anc2vec_data_generator import generator
from deepfold.models.esm_model import MLPLayer3D, MLPLayer
from deepfold.models.label_wise_attention import MultiHeadAttention
from torch.nn import BCEWithLogitsLoss

class Anc2vec(nn.Module):
    def __init__(self,one_hot,sub_ont_label,anc_label,latent_dim=256):
        super().__init__()
        self.nb_classes = one_hot.shape[0]
        self.latent_dim = latent_dim
        self.one_hot = one_hot.float()
        self.sub_ont_label = sub_ont_label.float()
        self.anc_label = anc_label.float()
        self.W = nn.Linear(self.nb_classes,self.latent_dim)
        self.R = nn.Linear(self.latent_dim,self.nb_classes)
        self.S = nn.Linear(self.latent_dim,3)
        self.A = nn.Linear(self.latent_dim,self.nb_classes)
    
    def forward(self):
        H = self.W(self.one_hot)
        bce_fn = BCEWithLogitsLoss()
        # reconstruct loss
        reconstruct_pred = self.R(H)
        reconstruct_loss = bce_fn(reconstruct_pred,self.one_hot)
        # sub ontology loss
        sub_ont_pred = self.S(H)
        sub_ont_loss = bce_fn(sub_ont_pred,self.sub_ont_label)
        # ancessor loss
        anc_pred = self.A(H)
        anc_loss = bce_fn(anc_pred,self.anc_label)

        return H, reconstruct_loss + sub_ont_loss + anc_loss


class P2GO(nn.Module):
    def __init__(self,
                 one_hot,sub_ont_label,anc_label,adj,
                 model_dir: str = 'esm1b_t33_650M_UR50S',
                 aa_dim=1280,
                 latent_dim=256,
                 dropout_rate=0.1,
                 n_head=2):
        super().__init__()
        self.one_hot = one_hot
        self.sub_ont_label = sub_ont_label
        self.anc_label =anc_label
        self.latent_dim = latent_dim
        self.adj = adj
        
        # backbone
        self.backbone, _ = esm.pretrained.load_model_and_alphabet(
            model_dir)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.nb_classes = adj.shape[0]
        # AA_emebdding transform
        self.aa_transform = MLPLayer3D(aa_dim, latent_dim)
        # go embedder
        self.anc2vec_model = Anc2vec(self.one_hot,self.sub_ont_label,
                                self.anc_label,self.latent_dim)

        # label-wise attention
        self.attention = MultiHeadAttention(latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            num_heads=n_head,
                                            dropout=dropout_rate)
        # gnn module
        self.gcn = GCN(latent_dim, latent_dim)
        # output layer
        self.go_transform_post = MLPLayer(int(2 * latent_dim), latent_dim)

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
        #     mean = x[i, :length].mean(axis=0, keepdim=True)
        #     mean_list.append(mean)
        # mean_batch = torch.cat(mean_list)
        # mean_embedding = mean_batch.unsqueeze(1)
        # mean_embedding = mean_embedding.repeat((1, self.nb_classes, 1))
        # go embedder
        # go_embedding [nb_classes,latent_dim]
        go_embedding, go_loss = self.anc2vec_model()
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
        outputs = logits
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))
            loss = loss + go_loss
            outputs = (logits,loss)
        if output_attention_weights:
            outputs = (loss, logits, weights)

        return outputs


if __name__ == '__main__':
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
    one_hot, anc_label,sub_ont_label = generator(label_map,go_file)
    one_hot = one_hot.cuda()
    sub_ont_label = sub_ont_label.cuda()
    anc_label = anc_label.cuda()
    adj = adj.cuda()

    model = P2GO(one_hot=one_hot,sub_ont_label=sub_ont_label,anc_label=anc_label,adj=adj,latent_dim=256,n_head=2)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=1e-3)
    model = model.cuda()
    model.train()
    optimizer.zero_grad()


    outputs = model(**batch)
    loss = outputs[1]
    loss.backward()
    optimizer.step()
    print(f'logits:{outputs[0].shape}')
    print(f'loss:{loss}')
    print(f'attention weights:{outputs[2].shape}')
    # for index, batch in enumerate(mfo_loader):
    #     batch = {key: val.cuda() for key, val in batch.items()}
    #     optimizer.zero_grad()
    #     outputs = model(**batch)
    #     loss = outputs[1]
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())



