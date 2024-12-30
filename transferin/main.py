import numpy as np
from transferin.model import get_model
import torch
import matplotlib.pyplot as plt
from preprocessing.docking_score import get_protein_cords, EPS, get_mol_cords, calculate_dsw
from sklearn.metrics.pairwise import euclidean_distances

pdb_file = "transferin/transferrin.pdb"
protein_seq, protein_cords = get_protein_cords(pdb_file)
protein_cords = np.array(protein_cords)

sdf_file = ""
mol_cords = get_mol_cords(sdf_file)
all_mol_coords = np.array(mol_cords)
dist = euclidean_distances(protein_cords, all_mol_coords)
weights = calculate_dsw(dist)
weights = weights.mean(axis=1)
weights = weights / (weights.sum() + EPS)

model = get_model()
emb = np.load("transferin/emb.npy")
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
plt.ylim([0.95, 1])
plt.show()
