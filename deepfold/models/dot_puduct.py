import math
import torch
import torch.nn as nn
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


class DotProductPrediction(nn.Module):
    def __init__(self,one_hot,sub_ont_label,anc_label,
                 seq_dim: int = 1280,
                 node_dim: int = 256,
                 latent_dim: int = 512):
        super().__init__()

        
        self.one_hot = one_hot
        self.sub_ont_label = sub_ont_label
        self.anc_label = anc_label
        self.nb_classes = self.one_hot.shape[0]
        self.node_dim = node_dim
        self.latent_dim = latent_dim
        # go embedder
        self.anc2vec_model = Anc2vec(self.one_hot,self.sub_ont_label,
                                self.anc_label,self.node_dim)
        self.seq_trans = MLPLayer(seq_dim, self.latent_dim)
        self.node_trans = MLPLayer(self.node_dim, self.latent_dim)
        


    def forward(self, embeddings, labels):
        embeddings = self.seq_trans(embeddings)
        go_embedding, go_loss = self.anc2vec_model()
        go_embedding = self.node_trans(go_embedding)
        go_embedding = go_embedding.transpose(-2, -1)
        logits = torch.matmul(embeddings, go_embedding)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))
            loss = loss + go_loss
            outputs = (loss,logits)

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
    from deepfold.data.gcn_dataset import GCNDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import pandas as pd

    data_path = '../../data/cafa3/'
    go_file = '../../data/cafa3/go_cafa3.obo'
    terms_df = pd.read_pickle('../../data/cafa3/mfo/mfo_terms.pkl')
    label_map = {v:k for k,v in enumerate(terms_df.terms.values.flatten())}
    # CAFA3 esm dataset
    mfo_dataset = GCNDataset(label_map,root_path='../../data/cafa3',
                             file_name='mfo_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl')
    mfo_loader = DataLoader(mfo_dataset,
                            batch_size=8)
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
    from deepfold.utils.anc2vec_data_generator import generator
    
    one_hot, anc_label,sub_ont_label = generator(label_map,go_file)
    one_hot = one_hot.cuda()
    sub_ont_label = sub_ont_label.cuda()
    anc_label = anc_label.cuda()

    model = DotProductPrediction(one_hot=one_hot,sub_ont_label=sub_ont_label,anc_label=anc_label,seq_dim=1280,node_dim=256,latent_dim=512)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   model.parameters()),
                            lr=1e-4)
    model = model.cuda()
    model.train()
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs[1]
    loss.backward()
    optimizer.step()
    print(f'logits:{outputs[0].shape}')
    print(f'loss:{loss}')
    # for index, batch in enumerate(mfo_loader):
    #     batch = {key: val.cuda() for key, val in batch.items()}
    #     optimizer.zero_grad()
    #     outputs = model(**batch)
    #     loss = outputs[1]
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())



