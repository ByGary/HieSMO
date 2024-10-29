import time
import warnings
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from scipy.sparse import coo_matrix, csr_matrix
from preprocess import *
from utils import *
from graph_model import CombinedModel


device = torch.device('cuda:1')
cfg = load_config()
print(cfg)
result_folder = cfg.train.result_folder
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
save_code(result_folder)

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
    warnings.warn('You have choosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

################ Loading data ################
# read data
file_fold = ''
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

################ Pre-processing data (human lymph node) ################
# RNA-seq
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)

gene_names = adata_omics1_high.var_names
gene2id = dict(zip(gene_names, range(gene_names.shape[0])))
pro_names = adata_omics2.var_names
pro2id = dict(zip(pro_names, range(pro_names.shape[0])))

################ Constructing omics-specific graph ################
adata_omics1.uns['interaction_net'] = build_omics_adj(adata_omics1_high.var_names, '', gene2id)
adata_omics2.uns['interaction_net'] = build_omics_adj(adata_omics2.var_names, '', pro2id)

################ Constructing spatial graph for cell neighbors ################
adata_omics1_high.uns['spatial_net'], spatial_coo, wnn_coo = build_spot_adj(cfg, adata_omics1_high, adata_omics2, n_neighbors=cfg.GAT.n_neighbors)

################ define the model ################
epochs = cfg.train.epochs
best_loss = float('inf')
n_obs, n_gene, n_pro = adata_omics1_high.shape[0], adata_omics1_high.shape[1], adata_omics2.shape[1]
best_aff_matrices = torch.zeros([n_obs, n_gene, n_pro]).to(device, torch.float64)

results_df = pd.DataFrame(columns=['Epoch', 'Loss'])
writer = SummaryWriter(os.path.join(result_folder, 'runs'))

combined_model = CombinedModel(n_obs, n_gene, n_pro).to(device, torch.float64)

adata_omics1_high.obs['cell_type'] = load_label('')
label_CSL = add_contrastive_label(adata_omics1_high.shape[0]).to(device, torch.float64)  # 对比学习标签

optimizer = Adam(combined_model.parameters(), lr=cfg.train.lr)
scheduler = StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
loss_CSL = nn.BCEWithLogitsLoss().to(torch.float64)

print(combined_model)
train_begin = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    print("Epoch:{}/{}".format(epoch, epochs))
    combined_model.train()
    optimizer.zero_grad()
    hidden_emb, ret = combined_model(adata_omics1, adata_omics1_high, adata_omics2, spatial_coo, wnn_coo, device)
    loss = loss_CSL(ret, label_CSL)
    loss.backward()
    optimizer.step()
    scheduler.step()

    results_dict = {
        'Epoch': epoch,
        'Loss': loss.item()
    }
    results_df = pd.concat([results_df, pd.DataFrame([results_dict])], ignore_index=True)

    if loss < best_loss:
        best_loss = loss
        model_dict = {
            'combined_model': combined_model.state_dict()
        }
        torch.save(model_dict, os.path.join(result_folder, 'best.pth'))
        adata_omics1_high.obsm['hidden_emb'] = hidden_emb.detach().cpu().numpy()

    writer.add_scalar('Loss', loss.item(), epoch)

    print("Loss:{:.6f} Epoch duration:{:.2f} min".format(loss.item(), (time.time() - epoch_start_time) / 60))

results_df.to_csv(os.path.join(result_folder, 'results.csv'), index=False)
writer.close()
print("The training has been completed, taking {:.2} hours".format((time.time() - train_begin) / 3600))

print(adata_omics1_high.obsm['hidden_emb'])

adata_omics1_high_reduced = adata_omics1_high.copy()
adata_omics2_reduced = adata_omics2.copy()
del adata_omics1_high_reduced.X
del adata_omics2_reduced.X
adata_omics1_high_reduced.uns['spatial_net'] = csr_matrix(adata_omics1_high.uns['spatial_net'])
adata_omics1_high_reduced.uns['interaction_net'] = csr_matrix(adata_omics1.uns['interaction_net'])
adata_omics2_reduced.uns['interaction_net'] = csr_matrix(adata_omics2.uns['interaction_net'])

sc.write(os.path.join(result_folder, 'adata_omics1_high.h5ad'), adata_omics1_high_reduced)
sc.write(os.path.join(result_folder, 'adata_omics2.h5ad'), adata_omics2_reduced)


if cfg.plt.tool == 'mclust':
    clustering(adata_omics1_high, cfg.plt.n_clusters, method=cfg.plt.tool)
elif cfg.plt.tool in ['leiden', 'louvain']:
    clustering(adata_omics1_high, cfg.plt.n_clusters, method=cfg.plt.tool, start=cfg.plt.start, end=cfg.plt.end, increment=cfg.plt.increment)

plt.rcParams["figure.figsize"] = (8, 6)
sc.tl.umap(adata_omics1_high)
sc.pl.umap(adata_omics1_high, color=[cfg.plt.tool], show=False)
plt.savefig(os.path.join(result_folder, 'UMAP.png'))
plt.show()

plt.rcParams["figure.figsize"] = (8, 6)
adata_omics1_high.obsm["spatial"][:, 0] *= 2
adata_omics1_high.obsm['spatial'][:, 1] *= -1
sc.pl.spatial(adata_omics1_high, color=[cfg.plt.tool, 'cell_type'], spot_size=2.2, show=False)
plt.savefig(os.path.join(result_folder, cfg.plt.tool + '.png'))
plt.show()

MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity = calculate_clustering_metrics(adata_omics1_high.obs['cell_type'], adata_omics1_high.obs[cfg.plt.tool])
with open(os.path.join(result_folder, 'clustering_metrics.txt'), 'w') as f:
    f.write(f"MI: {MI:.6f}\n")
    f.write(f"NMI: {NMI:.6f}\n")
    f.write(f"AMI: {AMI:.6f}\n")
    f.write(f"RI: {RI:.6f}\n")
    f.write(f"ARI: {ARI:.6f}\n")
    f.write(f"Homogeneity: {Homogeneity:.6f}\n")
    f.write(f"completeness: {completeness:.6f}\n")
    f.write(f"V_measure: {V_measure:.6f}\n")
    f.write(f"Purity: {purity:.6f}\n")
print("MI:{} NMI:{} AMI:{} RI:{} ARI:{} Homogeneity:{} completeness:{} V_measure:{} purity:{}".format(MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity))