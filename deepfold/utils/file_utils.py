import logging
import os
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_embedding(data_path, split='train'):
    datasetFolderPath = data_path
    trainFilePath = os.path.join(
        datasetFolderPath, 'esm1b_t33_650M_UR50S_embeddings_mean_train.pkl')
    testFilePath = os.path.join(
        datasetFolderPath, 'esm1b_t33_650M_UR50S_embeddings_mean_test.pkl')
    if split == 'train':
        data_df = pd.read_pickle(trainFilePath)
    else:
        data_df = pd.read_pickle(testFilePath)

    embeddings = list(data_df['esm_embeddings'])
    proteins = list(data_df['proteins'])

    emb_dict = [(protein, np.array(embedding))
                for protein, embedding in zip(proteins, embeddings)]

    return emb_dict


def save_from_generator(
    emb_path: str,
    the_generator: List[Tuple[str, np.ndarray]],
):
    if emb_path.endswith('.h5'):
        with h5py.File(str(emb_path), 'w') as hf:
            for sequence_id, embedding in the_generator:
                if emb_path.endswith('.h5'):
                    # noinspection PyUnboundLocalVariable
                    hf.create_dataset(sequence_id, data=embedding)
    elif emb_path.endswith('.npz'):
        emb_dict = dict()
        for sequence_id, embedding in the_generator:
            if embedding is None:
                # The generator code already showed an error
                continue
            emb_dict[sequence_id] = embedding

        if not emb_dict:
            raise RuntimeError('Embedding dictionary is empty!')
        logger.info('Total number of embeddings: {}'.format(len(emb_dict)))

        logger.info(f'Writing embeddings to {emb_path}')
        # With checked that the endswith can only be .npz
        np.savez(emb_path, **emb_dict)
    else:
        raise RuntimeError(
            f'The output file must end with .npz or .h5,'
            f"but the path you provided ends with '{emb_path[:-10]}'")


def save_annotations(data_path):
    datasetFolderPath = data_path
    trainFilePath = os.path.join(datasetFolderPath, 'train_data.pkl')
    data_df = pd.read_pickle(trainFilePath)
    save_path = os.path.join(data_path, 'train_annotations.txt')
    df = data_df[['proteins', 'exp_annotations']]
    with open(save_path, 'w') as f:
        for row in df.itertuples():
            f.write(row.proteins + ' ')
            for anno in row.exp_annotations:
                f.write(anno + ',')
            f.write('\n')


def read_embeddings(embeddings_in):
    """Read embeddings from h5 file generated by bio_embeddings pipeline.

    :param embeddings_in:
    :return:
    """
    embeddings = dict()
    with h5py.File(embeddings_in, 'r') as f:
        for sequence_id, embedding in f.items():
            embeddings[sequence_id] = np.array(embedding)
    return embeddings


if __name__ == '__main__':
    data_path = '/home/niejianzheng/xbiome/datasets/protein'
    # emb_dict = load_embedding(data_path, 'test')
    # emb_path = os.path.join(data_path, 'test_emb.h5')
    # print(emb_path)
    # save_from_generator(emb_path, emb_dict)

    # emb_dict = load_embedding(data_path, 'train')
    # emb_path = os.path.join(data_path, 'train_emb.h5')
    # print(emb_path)
    # save_from_generator(emb_path, emb_dict)
    save_annotations(data_path)
