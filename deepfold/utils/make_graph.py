import os


import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from deepfold.data.utils.ontology import Ontology
from deepfold.utils.make_edges import get_go_ic



def load_edge_list(all_go_ic, symmetrize=False):
    sorted_go_ic = sorted(all_go_ic, key=lambda k: (k[0], k[1]))
    df = pd.DataFrame(sorted_go_ic)
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={0: 1, 1: 0})
        df = pd.concat([df, rev])
    idx, _ = pd.factorize(df[[0, 1]].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    IC = df[2].astype('float')
    return idx, IC.tolist()


def load_node_list(all_go_ic):
    go_id = []
    for item in all_go_ic:
        tmp = [int(item[0][3:]), int(item[1][3:])]
        go_id.append(tmp)
    return go_id


def build_adj(idx, IC, idx_map):
    adj_new = []
    for i in range(len(idx_map)):
        tmp = [0] * (len(idx_map))
        adj_new.append(tmp)

    for i in range(len(idx)):
        adj_new[idx[i][0]][idx[i][1]] = IC[i]
        # 为什么要两次？
        adj_new[idx[i][1]][idx[i][0]] = IC[i]
    adj_new = np.array(adj_new)
    return adj_new


def normalize(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_adj):
    sparse_adj = sparse_adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_adj.row, sparse_adj.col)).astype(np.int64))
    values = torch.from_numpy(sparse_adj.data)
    shape = torch.Size(sparse_adj.shape)
    return torch.sparse.LongTensor(indices, values, shape)

def multi_hot_encoding(label_map, label_map_ivs, go_file):
    go_ont = Ontology(go_file)
    multi_hot = []
    go = {}
    for term in label_map.keys():
        if term not in label_map.keys():
            continue
        ancestors = set(go_ont.get_ancestors(term))
        ancestors_namespace = ancestors.intersection(set(label_map.keys()))
        tmp = [0] * len(label_map)
        if len(ancestors_namespace) == 1:
            index = label_map[term]
            tmp[index] = 1
        else:
            for ancestor in ancestors:
                index = label_map[ancestor]
                tmp[index] = 1
        go[term] = tmp
    for k in range(len(label_map_ivs)):
        tmp_go = label_map_ivs[k]
        if tmp_go in go.keys():
            multi_hot.append(go[tmp_go])
    multi_hot = torch.FloatTensor(multi_hot)
    return multi_hot


def build_graph(data_path='.data/', go_file_name='go_cafa3.obo', namespace='bpo'):
    go_file = os.path.join(data_path, go_file_name)
    all_edge_weights = get_go_ic(namespace=namespace, go_file_name=go_file_name,data_path=data_path)
    print(f'number of edges:{len(all_edge_weights)}')
    idx, IC = load_edge_list(all_edge_weights, symmetrize=False)
    idx_2d = idx
    idx = idx.reshape(-1)

    # convert to 1-dim array
    go_id = load_node_list(all_edge_weights)
    go_id = np.array(go_id)
    go_id = go_id.reshape(-1)

    # remove duplicate values
    go_id = pd.unique(go_id).tolist()
    go_id = sorted(go_id)
    # print(go_id)
    tmp = []
    idx = pd.unique(idx).tolist()
    for i in range(len(go_id)):
        go_id[i] = format(go_id[i], '07')
        tmp.append('GO:%07d' % (int(go_id[i])))

    # making dictionary
    idx_map = dict(zip(go_id, idx))
    label_map = dict(zip(tmp, idx))
    label_map_ivs = dict(zip(idx, tmp))

    # build symmetric adjacency matrix
    # why build symmetric adj?
    adj = build_adj(idx_2d, IC, idx_map)
    adj = sp.coo_matrix(adj)
    # adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(
    #     adj, (adj.T > adj))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    multi_hot_vector = multi_hot_encoding(label_map, label_map_ivs, go_file)

    return adj, multi_hot_vector, label_map, label_map_ivs,idx_2d


if __name__ == '__main__':
    # multi-hot
    data_path = '../../data/curated'
    go_file = os.path.join(data_path, 'go.obo')
    for namespace in ['cco','bpo', 'mfo']:  # 'bpo', 'mfo'
        print('---' * 5 + namespace + '---' * 5)
        adj, multi_hot_vector, label_map, label_map_ivs,idx_2d = build_graph(
            data_path=data_path, namespace=namespace)
        print(adj.shape)
        print(multi_hot_vector.shape)
        print(len(label_map))
        print(len(label_map_ivs))
