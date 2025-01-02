import numpy as np
from transferin.f_model import get_model
import torch
import matplotlib.pyplot as plt
from preprocessing.docking_score import get_protein_cords, EPS, get_mol_cords, calculate_dsw
from sklearn.metrics.pairwise import euclidean_distances

pdb_file = "transferin/transferrin.pdb"
protein_seq, protein_cords = get_protein_cords(pdb_file)
protein_cords = np.array(protein_cords)

names = ['DPPC', 'colesterol', 'DSPE-PEG1000', 'DSPE-PEG2000']
rartio = [0.65, 0.3, 0.025, 0.025]

all_weights = []
sdf_files = []
for name, r in zip(names, rartio):
    sdf_file = f"transferin/{name}.sdf"
    sdf_files.append(sdf_file)
    file_mol_cords = get_mol_cords(sdf_file)  # shape (n, 3)
    all_mol_coords = np.array(file_mol_cords)
    dist = euclidean_distances(protein_cords, all_mol_coords)
    w = calculate_dsw(dist)
    w = w.mean(axis=1)
    w = w / (w.sum() + EPS)
    plt.plot(w)
    all_weights.append(w)
plt.show()

weights = np.average(all_weights, axis=0, weights=rartio)

weights = torch.tensor(weights).float()

weights = weights.unsqueeze(1)

from train import get_tokenizer_and_model
from utils import ECType
from transformers import T5Config
from model import CustomT5Model

ec_type = ECType(2)
daa_type = 4
emb_dropout = 0
add_ec_tokens = 1
prot_dim = 2560
daa_model = CustomT5Model(T5Config(vocab_size=1091), daa_type, emb_dropout=emb_dropout, prot_dim=prot_dim)
print(f"Number of parameters: {sum(p.numel() for p in daa_model.parameters()):,}")
daa_model.load_state_dict(torch.load("transferin/daa_model.bin", map_location=torch.device('cpu')))
print(daa_model)
emb = np.load("transferin/emb_6B.npy")
if len(emb.shape) == 3:
    emb = emb[0]
emb = torch.tensor(emb).float()
x = emb.unsqueeze(0)
batch_size, seq_len, _ = x.size()
Q = daa_model.docking_attention.q_proj(x).view(batch_size, seq_len, daa_model.docking_attention.num_heads,
                                               daa_model.docking_attention.head_dim).transpose(1, 2)
K = daa_model.docking_attention.k_proj(x).view(batch_size, seq_len, daa_model.docking_attention.num_heads,
                                               daa_model.docking_attention.head_dim).transpose(1, 2)
V = daa_model.docking_attention.v_proj(x).view(batch_size, seq_len, daa_model.docking_attention.num_heads,
                                               daa_model.docking_attention.head_dim).transpose(1, 2)
attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (daa_model.docking_attention.head_dim ** 0.5)
x_attention = attn_weights[0].mean(dim=0)[0][1:-1]
print(attn_weights.shape)
print(weights.shape)
x_mean = 1 / len(attn_weights)
x_docking = weights.flatten()

res = daa_model.docking_attention.alpha * x_mean + daa_model.docking_attention.beta * x_docking + x_attention
res=res.detach().unsqueeze(1)
model = get_model()
# if 3D choose first in batch ([0])
emb_mean = emb.mean(0).unsqueeze(0)
res_mean = torch.nn.functional.sigmoid(model(emb_mean))[0].item()

print(res_mean)
seq_res = []
for seq_indes in range(emb.shape[0]):
    emb_seq = emb[seq_indes].unsqueeze(0)
    res_seq = torch.nn.functional.sigmoid(model(emb_seq))
    seq_res.append(res_seq[0].item())

w_emb = (emb[1:-1] * res).sum(0).unsqueeze(0)
emb_with_weights = torch.nn.functional.sigmoid(model(w_emb))[0].item()

print(emb_with_weights)

from transferin.pymol_viz.protein_values import create_pymol_script
from transferin.pymol_viz.protein_mols import create_pymol_script_with_sdf

values = np.array(seq_res[1:-1])
values = np.log((1 - values) / (values))

values = torch.Tensor(values)

values = torch.nn.functional.softmax(torch.Tensor(values)).numpy()
fig, ax = plt.subplots(figsize=(10, 3))
n_smooth = 5
v_plot = np.convolve(values, np.ones(n_smooth) / n_smooth, mode='valid')

ax.plot(v_plot, label="BBBP Probability", c='b')
plt.legend(loc="upper left")

output_script = f"per_resid.pml"
create_pymol_script(
    pdb_file,
    values,
    output_script=output_script)

output_script = "mols.pml"
v = np.log(weights.numpy().flatten())
# v=np.sqrt(weights.numpy().flatten())
create_pymol_script_with_sdf(pdb_file,
                             sdf_files, v, output_script)
v_plot = np.convolve(v, np.ones(n_smooth) / n_smooth, mode='valid')

ax.twinx().plot(v_plot, label="DAA", color="g")

ax.set_ylabel("BBBP Probability / DAA Weight")
ax.set_xlabel("Residue Index")
#remove the x,y ticks
plt.xticks([])
plt.yticks([])
plt.legend(loc="upper right")

plt.savefig("transferin/lines.png", dpi=300)





