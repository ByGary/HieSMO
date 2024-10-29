import glob
import os
import random
import shutil
import numpy as np
import torch
from scipy.sparse import coo_matrix
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, rand_score, adjusted_rand_score, \
    homogeneity_score, completeness_score, v_measure_score, confusion_matrix
import libpysal
import esda
from scipy.spatial.distance import jaccard
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from torch.backends import cudnn

from config.defaults import get_default_config


def load_config():
    config = get_default_config()
    config.merge_from_file('config/exp.yaml')
    config.freeze()
    return config


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


def save_code(result_folder):

    source_dir = os.getcwd()
    target_dir = os.path.join(result_folder, 'code')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    python_files = glob.glob(os.path.join(source_dir, '*.py'))
    for file_path in python_files:
        shutil.copy(file_path, target_dir)
    print(f"All .py files have been copied to {target_dir}")

    shutil.copy(os.path.join(source_dir, 'config', 'exp.yaml'), os.path.join(result_folder, 'exp.yaml'))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def calculate_clustering_metrics(y_true, y_pred):
    MI = mutual_info_score(y_true, y_pred)
    NMI = normalized_mutual_info_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    RI = rand_score(y_true, y_pred)
    ARI = adjusted_rand_score(y_true, y_pred)
    Homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    V_measure = v_measure_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)

    return MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity


def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    return purity


def clustering(adata, n_clusters=10, key='hidden_emb', method='mclust', start=0.1, end=2.0, increment=0.01):

    if method == 'mclust':
        sc.pp.neighbors(adata, n_neighbors=20, use_rep=key)
        adata = mclust_R(adata, num_cluster=n_clusters, used_obsm=key)
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)


def mclust_R(adata, num_cluster, used_obsm='hidden_emb', modelNames='EEE'):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print("res:", res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def search_res(adata, n_clusters, method='leiden', use_rep='hidden_emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    best_ari, best_res = 0.0, 0.0
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity = calculate_clustering_metrics(adata.obs['cell_type'], adata.obs[cfg.plt.tool])
            if ARI > best_ari and count_unique == n_clusters:
                best_ari = ARI
                best_res = res
            print('resolution={}, cluster number={}, ARI={}, AMI={} NMI={}, Homogeneity={} completeness={} V_measure={} purity={}'.format(res, count_unique, ARI, AMI, NMI, Homogeneity, completeness, V_measure, purity))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            # break
    print("best_ari:{} best_res:{}".format(best_ari, best_res))
    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!"

    return best_res

