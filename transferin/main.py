import numpy as np
from transferin.model import get_model
import torch
import matplotlib.pyplot as plt
from preprocessing.docking_score import get_protein_cords, EPS, get_mol_cords, calculate_dsw
from sklearn.metrics.pairwise import euclidean_distances

pdb_file = "transferin/transferrin.pdb"
protein_seq, protein_cords = get_protein_cords(pdb_file)
protein_cords = np.array(protein_cords)

names = ['DPPC', 'colesterol', 'DSPE-PEG1000', 'DSPE-PEG2000']
rartio = [0.65, 0.3, 0.025, 0.025]

mol_cords = []
all_weights = []
for name, r in zip(names, rartio):
    sdf_file = f"transferin/{name}.sdf"
    file_mol_cords = get_mol_cords(sdf_file)  # shape (n, 3)
    mol_cords.extend(file_mol_cords)
    all_mol_coords = np.array(mol_cords)

    dist = euclidean_distances(protein_cords, all_mol_coords)
    w = calculate_dsw(dist)
    w = w.mean(axis=1)
    w = w / (w.sum() + EPS)
    all_weights.append(w)
weights = np.average(all_weights, axis=0, weights=rartio)

weights = torch.tensor(weights).float()

weights = weights.unsqueeze(1)

model = get_model()
emb = np.load("transferin/emb_6B.npy")
# if 3D choose first in batch ([0])
if len(emb.shape) == 3:
    emb = emb[0]
emb = torch.tensor(emb).float()
emb_mean = emb.mean(0).unsqueeze(0)
res_mean = torch.nn.functional.sigmoid(model(emb_mean))[0].item()

print(res_mean)
seq_res = []
for seq_indes in range(emb.shape[0]):
    emb_seq = emb[seq_indes].unsqueeze(0)
    res_seq = torch.nn.functional.sigmoid(model(emb_seq))
    seq_res.append(res_seq[0].item())

plt.plot(seq_res)
# add hline with mean
plt.axhline(y=res_mean, color='r', linestyle='-')

# for alpha in [0, 0.1, 0.2, 0.5, 0.9, 1]:
w_emb = (emb[1:-1] * weights).sum(0).unsqueeze(0)
emb_with_weights = torch.nn.functional.sigmoid(model(w_emb))[0].item()
print(emb_with_weights)
plt.axhline(y=emb_with_weights, linestyle='--', color='g')
# plt.ylim([0.95, 1])
# plt.show()
from transferin.pymol_viz.protein_values import create_pymol_script
from transferin.pymol_viz.utils import replace_local_pathes

seq_res = np.array(seq_res)
output_script = f"transferin.pymol_viz/per_resid.pml"
create_pymol_script(
    pdb_file,
    seq_res,
    output_script=output_script)
replace_local_pathes(output_script)

