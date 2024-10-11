########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
# https://github.com/JK-Liu7/AttentionMGT-DTA
# https://github.com/595693085/DGraphDTA

########################################################################################################################
########## Import
########################################################################################################################

# import itertools
# from collections import defaultdict
# from operator import neg
# import math
# import json
# from rdkit.Chem import AllChem
# import pandas as pd

import os
import torch
import random
import numpy as np
from rdkit import Chem
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


########################################################################################################################
########## Functions - Common
########################################################################################################################

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        # self.num_nodes = x_s.size(0) + x_t.size(0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class CustomDataset(Dataset):
    def __init__(self, tri_list , mode, shuffle=False):
        self.tri_list = tri_list
        self.mode = mode

        if shuffle:
            random.shuffle(self.tri_list)

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):
        d1_samples = []
        d2_samples = []
        bi_samples = []
        labels = []
        for d1, d2, label in batch:
            d1_graph = self.__create_graph_data(d1, self.mode['d1'])
            d2_graph = self.__create_graph_data(d2, self.mode['d2'])

            bi_edge_index = get_bipartite_graph(len(d1_graph.x), len(d2_graph.x))
            bi_graph = self._create_b_graph(bi_edge_index, d1_graph.x, d2_graph.x)

            d1_samples.append(d1_graph)
            d2_samples.append(d2_graph)
            bi_samples.append(bi_graph)
            labels.append(label)

        d1_samples = Batch.from_data_list(d1_samples)
        d2_samples = Batch.from_data_list(d2_samples)
        bi_samples = Batch.from_data_list(bi_samples)
        if self.mode['task'] == 'DTA':
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return d1_samples, d2_samples, labels, bi_samples

    def __create_graph_data(self, m, mode):
        if mode == 'Drug':
            edge_index, n_features = get_mol_edge_list_and_feat_mtx(m)
        elif mode == 'Protein':
            edge_index, n_features = target_to_graph(m)
        else:
            raise Exception('Input type error!!!')

        return Data(x=n_features, edge_index=edge_index)

    def _create_b_graph(self, edge_index, x_s, x_t):
        return BipartiteData(edge_index, x_s, x_t)


class CustomDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bipartite_graph(graph_1, graph_2):
    x1 = np.arange(0, graph_1)
    x2 = np.arange(0, graph_2)
    edge_list = torch.LongTensor(np.array(np.meshgrid(x1,x2)))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list



########################################################################################################################
########## Functions - Drug
########################################################################################################################

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)



def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    return undirected_edge_list.T, n_features


########################################################################################################################
########## Functions - Protein
########################################################################################################################

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']


pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return torch.from_numpy(feature)


# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_inform):
    target_key, target_sequence, contact_dir, aln_dir = target_inform
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)

    # remove self-loop
    target_edge_index = torch.LongTensor([(i, j) for i, j in zip(index_row, index_col) if i != j]).T
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_feature = F.pad(target_feature, (0, 1), mode='constant', value=0)
    target_feature = target_feature.to(torch.float)
    return target_edge_index, target_feature


########################################################################################################################
########## Run
########################################################################################################################
