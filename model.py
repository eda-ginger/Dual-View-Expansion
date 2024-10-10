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
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    LayerNorm,
    global_add_pool
)

from layers import (
                    CoAttentionLayer,
                    RESCAL,
                    IntraGraphAttention,
                    InterGraphAttention,
                    )

########################################################################################################################
########## Model
########################################################################################################################


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.kge_dim)

    def forward(self, triples):
        data1, data2, labels, b_graph = triples

        data1.x = self.initial_norm(data1.x, data1.batch)
        data2.x = self.initial_norm(data2.x, data2.batch)
        repr_d1 = []
        repr_d2 = []

        for i, block in enumerate(self.blocks):
            out = block(data1, data2, b_graph)

            data1 = out[0]
            data2 = out[1]
            r_d1 = out[2]
            r_d2 = out[3]
            repr_d1.append(r_d1)
            repr_d2.append(r_d2)

            data1.x = F.elu(self.net_norms[i](data1.x, data1.batch))
            data2.x = F.elu(self.net_norms[i](data2.x, data2.batch))

        repr_d1 = torch.stack(repr_d1, dim=-2)
        repr_d2 = torch.stack(repr_d2, dim=-2)

        attentions = self.co_attention(repr_d1, repr_d2)
        scores = self.KGE(repr_d1, repr_d2, attentions)
        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, data1, data2, b_graph):
        data1.x = self.feature_conv(data1.x, data1.edge_index)
        data2.x = self.feature_conv(data2.x, data2.edge_index)

        d1_intraRep = self.intraAtt(data1)
        d2_intraRep = self.intraAtt(data2)

        d1_interRep, d2_interRep = self.interAtt(data1, data2, b_graph)

        d1_rep = torch.cat([d1_intraRep, d1_interRep], 1)
        d2_rep = torch.cat([d2_intraRep, d2_interRep], 1)
        data1.x = d1_rep
        data2.x = d2_rep

        # readout
        d1_att_x, att_edge_index, att_edge_attr, d1_att_batch, att_perm, d1_att_scores = self.readout(data1.x,
                                                                                                   data1.edge_index,
                                                                                                   batch=data1.batch)
        d2_att_x, att_edge_index, att_edge_attr, d2_att_batch, att_perm, d2_att_scores = self.readout(data2.x,
                                                                                                   data2.edge_index,
                                                                                                   batch=data2.batch)

        d1_global_graph_emb = global_add_pool(d1_att_x, d1_att_batch)
        d2_global_graph_emb = global_add_pool(d2_att_x, d2_att_batch)

        return data1, data2, d1_global_graph_emb, d2_global_graph_emb


