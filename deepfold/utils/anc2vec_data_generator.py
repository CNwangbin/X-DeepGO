import torch
from deepfold.data.utils.ontology import Ontology


def generator(label_map,go_file):
    terms = list(label_map.keys())
    nb_classes = len(terms)
    one_hot = torch.eye(nb_classes)
    go_rel = Ontology(go_file,True)
    anc_label = [[0]*nb_classes]*nb_classes
    sub_ont_label = [[0]*3]*nb_classes
    namespace_dict = {
            'biological_process': 0,
            'cellular_component': 1,
            'molecular_function': 2
        }
    for i in range(nb_classes):
        term = terms[i]
        anchestors = list(go_rel.get_ancestors(term))
        for j in range(len(anchestors)):
            anchestor = anchestors[j]
            if anchestor in terms:
                anc_label[i][j] = 1
        name = go_rel.get_namespace(term)
        name_idx = namespace_dict[name]
        sub_ont_label[i][name_idx] = 1
    return one_hot,torch.Tensor(anc_label),torch.Tensor(sub_ont_label)


if __name__ == '__main__':
    from deepfold.utils.make_graph import build_graph
    data_path = '../../data/cafa3/'
    go_file = '../../data/cafa3/go_cafa3.obo'
    namespace = 'cco'
    adj, multi_hot_vector, label_map, label_map_ivs,_ = build_graph(
        data_path=data_path, namespace=namespace)
    one_hot, anc_label,sub_ont_label = generator(label_map,go_file)
    print(one_hot.shape)
    print(anc_label.shape)
    print(sub_ont_label.shape)

