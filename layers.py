########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction

########################################################################################################################
########## Import
########################################################################################################################

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

########################################################################################################################
########## Layers
########################################################################################################################

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k      # (#block, 55) @ (55, 27) -> (#block, 27)
        queries = attendant @ self.w_q  # (#block, 55) @ (55, 27) -> (#block, 27)

        # (#block, #block, 27)
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias

        # (#block, #block, 27) @ (27, ) -> (#block, #block) -> (4, 4)
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, heads, tails, alpha_scores):
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        scores = heads @ tails.transpose(-2, -1) # (batch, 4, 128) @ (batch, 128 @ 4) -> (batch, 4, 4)

        if alpha_scores is not None:
            scores = alpha_scores * scores # batch, 4, 4
        # scores = scores.sum(dim=(-2, -1))  # batch, 1
        scores = scores.sum(dim=-2) # batch, 4, 1 (sum row)
        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_features})"


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim, 32, 2)

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        return intra_rep


# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
        return h_rep, t_rep


