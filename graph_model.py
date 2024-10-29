import random
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union
from torch.backends import cudnn
from torch.nn import Parameter
from tqdm import tqdm
from gat_conv import GATConv
import numpy as np
from preprocess import permutation, transfer_pyG_omicsGraph, transfer_pyG_spatialGraph
from utils import load_config


cfg = load_config()
# to be reproducible
random_seed = cfg.train.seed
if torch.cuda.is_available():
    cudnn.benchmark = True
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.manual_seed(random_seed)  # Sets the seed for generating random numbers.
    torch.cuda.manual_seed(random_seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(random_seed)  # Sets the seed for generating random numbers on all GPUs.
    cudnn.deterministic = True


class CombinedModel(nn.Module):
    def __init__(self, n_obs, n_gene, n_pro):
        super(CombinedModel, self).__init__()
        self.n_obs = n_obs
        self.n_gene = n_gene
        self.n_pro = n_pro
        self.typing_graph = Typing_Garph()
        self.attention = AttentionLayer(self.n_gene, self.n_pro, cfg.GAT.in_dim)
        self.cgm = CGM()

    def forward(self, adata_omics1, adata_omics1_high, adata_omics2, spatial_coo, wnn_coo, device):
        cell_emb_gene = torch.zeros([self.n_obs, self.n_gene, cfg.GM.feat_dim]).to(torch.float64)
        cell_emb_protein = torch.zeros([self.n_obs, self.n_pro, cfg.GM.feat_dim]).to(torch.float64)
        for i in tqdm(range(self.n_obs)):
            node_features_gene, adj_gene = transfer_pyG_omicsGraph(adata_omics1_high, adata_omics1.uns['interaction_net'], i)
            node_features_pro, adj_pro = transfer_pyG_omicsGraph(adata_omics2, adata_omics2.uns['interaction_net'], i)
            cell_emb_gene[i], cell_emb_protein[i] = self.cgm(node_features_gene.to(device, torch.float64), adj_gene.to(device, torch.float64), node_features_pro.to(device, torch.float64), adj_pro.to(device, torch.float64))
        cell_emb_gene = torch.mean(cell_emb_gene, dim=2, keepdim=False)
        cell_emb_protein = torch.mean(cell_emb_protein, dim=2, keepdim=False)
        fused_feat = self.attention(cell_emb_gene.to(device), cell_emb_protein.to(device))
        spatial_data, spatial_list, wnn_list = transfer_pyG_spatialGraph(fused_feat.cpu(), adata_omics1_high.uns['spatial_net'], spatial_coo,wnn_coo, device)
        hidden_emb, ret = self.typing_graph(spatial_data, spatial_list, wnn_list)

        return hidden_emb, ret


class CGM(nn.Module):
    def __init__(self):
        super(CGM, self).__init__()
        self.gnn_layer = cfg.GM.gnn_layer
        self.feat_dim = cfg.GM.feat_dim

        for i in range(self.gnn_layer):
            if i == 0:
                encode_layer = Siamese_Gconv(1, self.feat_dim)
            else:
                encode_layer = Siamese_Gconv(self.feat_dim, self.feat_dim)
            self.add_module('encode_layer_{}'.format(i), encode_layer)

    def forward(self, feat_gene, A_gene, feat_pro, A_pro):

        for i in range(self.gnn_layer):
            encode_layer = getattr(self, 'encode_layer_{}'.format(i))
            feat_gene, feat_pro = encode_layer([feat_gene, A_gene], [feat_pro, A_pro])

        return feat_gene, feat_pro


class Typing_Garph(torch.nn.Module):
    def __init__(self):
        super(Typing_Garph, self).__init__()
        self.conv1 = GATConv(cfg.GAT.in_dim, cfg.GAT.num_hidden, heads=1, concat=True, add_self_loops=True, bias=True)
        self.conv2 = GATConv(cfg.GAT.num_hidden, cfg.GAT.out_dim, heads=1, concat=True, add_self_loops=True, bias=True)
        self.disc = Discriminator(cfg.GAT.out_dim)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, spatial_data, spatial_list, wnn_list):
        features = spatial_data.x
        edge_index = spatial_data.edge_index
        features_au = permutation(features)
        h1 = F.relu(self.conv1(features, wnn_list, attention=True))
        h2 = self.conv2(h1, spatial_list, attention=True)
        emb = F.relu(h2)

        h1_au = F.relu(self.conv1(features_au, wnn_list, attention=True))
        h2_au = self.conv2(h1_au, spatial_list, attention=True)
        emb_au = F.relu(h2_au)

        g = self.read(emb, edge_index)
        g = self.sigm(g)
        ret = self.disc(g, emb, emb_au)

        return h2, ret


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        n_spot = emb.shape[0]
        coo_mask = sp.coo_matrix((np.ones(mask.shape[1]), (mask[0, :].cpu(), mask[1, :].cpu())), shape=(n_spot, n_spot))
        coo_mask = torch.from_numpy(coo_mask.toarray()).to(emb.device, torch.float64)
        vsum = torch.mm(coo_mask, emb)
        row_sum = torch.sum(coo_mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class AttentionLayer(nn.Module):
    def __init__(self, n_gene, n_pro, fused_dim):
        super(AttentionLayer, self).__init__()
        self.project_gene = nn.Linear(n_gene, fused_dim)
        self.project_protein = nn.Linear(n_pro, fused_dim)

        self.w_omega = Parameter(torch.FloatTensor(fused_dim, fused_dim).to(torch.float64))
        self.u_omega = Parameter(torch.FloatTensor(fused_dim, 1).to(torch.float64))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, feat_gene, feat_pro):

        self.feat_gene = self.project_gene(feat_gene)
        self.feat_pro = self.project_protein(feat_pro)

        self.feat_gene = torch.unsqueeze(self.feat_gene, dim=1)
        self.feat_pro = torch.unsqueeze(self.feat_pro, dim=1)

        self.feat_cat = torch.cat([self.feat_gene, self.feat_pro], dim=1)

        self.v = F.tanh(torch.matmul(self.feat_cat, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1)

        emb_combined = torch.matmul(torch.transpose(self.feat_cat, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined)


class Siamese_Gconv(nn.Module):

    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1: [Tensor, Tensor], *args) -> Union[Tensor, List[Tensor]]:
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        else:
            returns = [emb1]
            for g in args:
                returns.append(self.gconv(*g))
            return returns


class Gconv(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(Gconv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.DoubleTensor(self.in_feat, self.out_feat))  # 双精度训练

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x
