########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
# https://github.com/595693085/DGraphDTA

########################################################################################################################
########## Import
########################################################################################################################

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

            d1_graph, d2_graph = self._zero_pad_nodes(d1_graph, d2_graph)

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
        elif self.mode['task'] == 'PPI':
            labels = torch.LongTensor(labels)
        else:
            raise Exception(f'Wrong task type {self.mode["task"]}')

        return d1_samples, d2_samples, labels, bi_samples

    def __create_graph_data(self, m, mode):
        if mode == 'Drug':
            edge_index, n_features = get_mol_edge_list_and_feat_mtx(m)
            return Data(x=n_features, edge_index=edge_index)
        elif mode == 'Protein':
            return m
        else:
            raise Exception('Input type error!!!')

    def _create_b_graph(self, edge_index, x_s, x_t):
        return BipartiteData(edge_index, x_s, x_t)

    def _zero_pad_nodes(self, d1, d2):
        max_dim = max(d1.x.shape[1], d2.x.shape[1])
        expand_d1 = max_dim - d1.x.shape[1]
        expand_d2 = max_dim - d2.x.shape[1]
        d1.x = F.pad(d1.x, (0, expand_d1, 0, 0), mode='constant', value=0)
        d2.x = F.pad(d2.x, (0, expand_d2, 0, 0), mode='constant', value=0)
        return d1, d2


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
########## Run
########################################################################################################################
