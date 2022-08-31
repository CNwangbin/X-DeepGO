import math
import torch
import torch.nn as nn
from deepfold.models.esm_model import MLPLayer3D, MLPLayer
from torch.nn import BCEWithLogitsLoss


class DotProductPrediction(nn.Module):
    def __init__(self,terms_embedding,
                 seq_dim: int = 1280,
                 latent_dim: int = 512,
                 loss_fn=None):
        super().__init__()
        self.terms_embedding = nn.Parameter(terms_embedding,requires_grad=True)
        self.node_dim = terms_embedding.shape[1]
        self.latent_dim = latent_dim
        self.nb_classes = terms_embedding.shape[0]
        self.seq_trans = MLPLayer(seq_dim, self.latent_dim)
        self.node_trans = MLPLayer(self.node_dim, self.latent_dim)
        self.loss_fn = loss_fn

    def forward(self, embeddings, labels):
        embeddings = self.seq_trans(embeddings)
        go_embedding = self.node_trans(self.terms_embedding)
        go_embedding = go_embedding.transpose(-2, -1)
        logits = torch.matmul(embeddings, go_embedding)
        outputs = (logits, )
        if labels is not None:
            if self.loss_fn is not None:
                loss_fct = self.loss_fn
            else:
                loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.nb_classes),
                            labels.view(-1, self.nb_classes))
            outputs = (loss,logits)

        return outputs


if __name__ == '__main__':
    from deepfold.utils.make_graph import build_graph
    from deepfold.data.gcn_dataset import GCNDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    import os

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
    
    terms_all = pd.read_pickle(os.path.join(data_path,'all_terms_partial_order_embeddings.pkl'))
    terms = pd.read_pickle(os.path.join(data_path,namespace,namespace + '_terms.pkl'))
    terms_embedding = terms.merge(terms_all)
    embeddings = np.concatenate([np.array(embedding,ndmin=2) for embedding in terms_embedding.embeddings.values])
    terms_embedding = torch.Tensor(embeddings)
    terms_embedding = terms_embedding.cuda()
    adj = adj.cuda()

    model = DotProductPrediction(terms_embedding=terms_embedding,seq_dim=1280,latent_dim=512)
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
    # for index, batch in enumerate(mfo_loader):
    #     batch = {key: val.cuda() for key, val in batch.items()}
    #     optimizer.zero_grad()
    #     outputs = model(**batch)
    #     loss = outputs[1]
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())



