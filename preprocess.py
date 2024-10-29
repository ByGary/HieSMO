import csv
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch.backends import cudnn
from torch_geometric.data import Data
from utils import load_config, sparse_mx_to_torch_sparse_tensor


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


def transfer_pyG_spatialGraph(cell_emb, spatial_net, spatial_coo, wnn_coo, device):
    edge_list = [spatial_net.row, spatial_net.col]
    data = Data(edge_index=torch.LongTensor(np.vstack([edge_list[0], edge_list[1]])).to(device),
                x=torch.DoubleTensor(cell_emb).to(device))
    spatial_list = torch.LongTensor(np.vstack([spatial_coo.row, spatial_coo.col])).to(device)
    wnn_list = torch.LongTensor(np.vstack([wnn_coo.row, wnn_coo.col])).to(device)
    return data, spatial_list, wnn_list


def transfer_pyG_omicsGraph(adata, coo_adj_matrix, cell_index):

    node_features = torch.FloatTensor(adata.X[cell_index, :].reshape(-1, 1))
    adj = sparse_mx_to_torch_sparse_tensor(coo_adj_matrix)

    return node_features, adj


def merge_omic_adj(gene_adj, omics_adj):

    row_list, col_list = [], []

    gene_indices = {(row, col) for row, col in zip(gene_adj.row, gene_adj.col)}
    omics_indices = {(row, col) for row, col in zip(omics_adj.row, omics_adj.col)}

    all_indices = gene_indices | omics_indices

    for (row, col) in all_indices:
        row_list.append(row)
        col_list.append(col)


    n = max(row_list) + 1
    coo_adj_matrix = sp.coo_matrix((np.ones(len(row_list)), (row_list, col_list)), shape=(n, n))
    coo_adj_matrix += sp.eye(coo_adj_matrix.shape[0], format='coo')
    coo_adj_matrix = coo_adj_matrix + coo_adj_matrix.T
    coo_adj_matrix.data = np.ones(coo_adj_matrix.nnz)

    return coo_adj_matrix


def build_KNN_adj(adata_omics):
    nbrs = kneighbors_graph(adata_omics.X.T, 1, mode="connectivity", metric="correlation", include_self=False)
    nbrs_coo = nbrs.tocoo()

    return nbrs_coo


def build_omics_adj(molecule_names, interaction_file, mole2id):

    row_list, col_list, value_list = [], [], []
    with open(interaction_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            node1, node2 = row[0], row[1]
            if node1 in molecule_names and node2 in molecule_names:
                row_list.append(node1)
                col_list.append(node2)

    row_series = pd.Series(row_list)
    col_series = pd.Series(col_list)

    node1_index = row_series.map(mole2id).tolist()
    node2_index = col_series.map(mole2id).tolist()

    n = len(mole2id)
    coo_adj_matrix = sp.coo_matrix((np.ones(len(node1_index)), (node1_index, node2_index)), shape=(n, n))
    return coo_adj_matrix


def build_spot_adj(cfg, adata_omics1_high, adata_omics2, n_neighbors=6):
    n = adata_omics1_high.X.shape[0]
    cell_position = adata_omics2.obsm['spatial']
    spatial_nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = spatial_nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    spatial_coo = coo_matrix((np.ones(len(x)), (list(x), list(y))), shape=(n, n), dtype=np.float64)

    wnn_idx = pd.read_csv("", header=0).values - 1
    wnn_dist = pd.read_csv("", header=0).values
    wnn_edge = (np.arange(n).repeat(wnn_idx.shape[1]), wnn_idx.flatten())
    wnn_coo = coo_matrix((1 / (wnn_dist.flatten() + 0.8), (wnn_edge[0], wnn_edge[1])), shape=(n, n), dtype=np.float64)

    union_coo = (spatial_coo + wnn_coo).tocoo()

    return union_coo, spatial_coo, wnn_coo


def permutation(cell_emb):
    ids = np.arange(cell_emb.shape[0])
    ids = np.random.permutation(ids)
    emb_permutated = cell_emb[ids]

    return emb_permutated


def add_contrastive_label(n_spot):
    # contrastive label
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = torch.tensor(np.concatenate([one_matrix, zero_matrix], axis=1))
    return label_CSL


def load_label(file):
    annotation_df = pd.read_csv(file, header=0)
    labels = annotation_df.iloc[:, 1].tolist()

    return labels


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata
