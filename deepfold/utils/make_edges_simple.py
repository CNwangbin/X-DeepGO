import math
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

from deepfold.data.utils.ontology import Ontology

sys.path.append('../')


# GOA_cnt
def statistic_terms(train_data_path):
    """get frequency dict from train file."""
    train_data = pd.read_pickle(train_data_path)
    cnt = Counter()
    for i, row in train_data.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1
    print('Number of annotated terms:', len(cnt))
    sorted_by_freq_tuples = sorted(cnt.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    freq_dict = {go: count for go, count in sorted_by_freq_tuples}
    return freq_dict


# make edges
def make_edges(go_file, namespace='bpo', with_rels=False,annotated_terms=None):
    go_ont = Ontology(go_file, with_rels=with_rels)
    if namespace == 'bpo':
        all_terms = go_ont.get_namespace_terms('biological_process')
    elif namespace == 'mfo':
        all_terms = go_ont.get_namespace_terms('molecular_function')
    elif namespace == 'cco':
        all_terms = go_ont.get_namespace_terms('cellular_component')
    
    if annotated_terms is not None:
        all_terms = all_terms.intersection(set(annotated_terms))
    edges = []
    for child in all_terms:
        parents = go_ont.get_parents(child)
        if len(parents) > 0:
            for parent in parents:
                if parent in all_terms:
                    edges.append((child, parent))
    return edges


# make IC file
def read_go_children(input_go_obo_file):
    children = defaultdict(list)
    alt_id = defaultdict(list)
    term = False
    go_id = ''
    alt_ids = set()
    with open(input_go_obo_file) as read_in:
        for line in read_in:
            splitted_line = line.strip().split(':')
            if '[Term]' in line:
                term = True
                go_id = ''
                alt_ids = set()
            elif term and 'id: GO:' in line and 'alt_id' not in line:
                go_id = 'GO:{}'.format(splitted_line[2].strip())
            elif term and 'alt_id: GO' in line:
                alt_id_id = 'GO:{}'.format(splitted_line[2].strip())
                alt_ids.add(alt_id_id)
                alt_id[go_id].append(alt_id_id)
            elif term and 'is_a:' in line:
                splitted_term = splitted_line[2].split('!')
                go_term = 'GO:{}'.format(splitted_term[0].strip())
                children[go_term].append(go_id)
                for a in alt_ids:
                    children[go_term].append(a)
            elif '[Typedef]' in line:
                term = False
    return children, alt_id

def make_ic_dict(train_data_file,go_file,terms):
    train_df = pd.read_pickle(train_data_file)
    try:
        annotations = train_df['annotations'].values
    except:
        annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    go_rels = Ontology(go_file, with_rels=True)
    go_rels.calculate_ic(annotations)

    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)
    return ics


# make final edge file
def get_edge_weight(edges, freq_dict, children_dict, ic_dict):
    all_go_cnt = []
    for child, parent in edges:
        ic_all = 0.0
        ic_child = 0.0
        cnt_freq_child = 0.0
        cnt_freq_parent = 0.0
        cnt_freq = 0.0

        # calculate prior P(Us|Ut)
        assert child in freq_dict.keys() and parent in freq_dict.keys()
        cnt_freq_child = freq_dict[child]
        cnt_freq_parent = freq_dict[parent]
        if cnt_freq_parent == 0.0 or cnt_freq_child == 0.0:
            cnt_freq = 1.0
        else:
            cnt_freq = cnt_freq_child / cnt_freq_parent
        # calculate IC
        if parent in children_dict:
            for x in children_dict[parent]:
                if x in ic_dict.keys():
                    ic_all += ic_dict[x]

        if child in ic_dict.keys():
            ic_child += ic_dict[child]
        if ic_all == 0:
            ic_score = 1.0
        else:
            ic_score = ic_child / ic_all
        final_cnt = ic_score  + cnt_freq
        all_go_cnt.append((parent, child, final_cnt))
    return all_go_cnt


def get_go_ic(namespace='bpo', data_path=None):
    go_file = os.path.join(data_path, 'go_cafa3.obo')
    train_data_file = os.path.join(data_path, namespace,
                                   namespace + '_train_data.pkl')
    freq_dict = statistic_terms(train_data_file)
    annotated_terms = freq_dict.keys()
    edges = make_edges(go_file, namespace,False,annotated_terms)
    children,alt_id = read_go_children(go_file)
    ic_dict = make_ic_dict(train_data_file,go_file,freq_dict.keys())
    all_weight = get_edge_weight(edges, freq_dict, children, ic_dict)
    return all_weight


if __name__ == '__main__':
    data_path = '../../data/cafa3'
    bpo_weight = get_go_ic('bpo', data_path)
    print(f'edges in bpo: {len(bpo_weight)}')
    # print(f'nodes in bpo:{}')
    mfo_weight = get_go_ic('mfo', data_path)
    print(f'edges in mfo: {len(mfo_weight)}')
    cco_weight = get_go_ic('cco', data_path)
    print(f'edges in cco: {len(cco_weight)}')
