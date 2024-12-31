import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from transferin.pymol_viz.utils import get_residue_ids_from_pdb, replace_local_pathes


def create_pymol_script(pdb_file: str, values: np.array, output_script):
    residue_ids = get_residue_ids_from_pdb(pdb_file)
    assert len(residue_ids) == len(values), "Number of residues in the PDB file does not match the number of values."
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_pca_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    viridis_colors = plt.cm.get_cmap("viridis")
    emb_colors = viridis_colors(normalized_pca_values)
    emb_colors = emb_colors[:, :3]
    with open(output_script, 'w') as f:
        # Load the PDB file in PyMOL
        f.write(f"load {pdb_file}, protein\n")
        for (chain_id, res_num), color in zip(residue_ids, emb_colors):
            r, g, b = color  # RGB values
            f.write(f"set_color color_{chain_id}_{res_num}, [{r}, {g}, {b}]\n")
            f.write(f"color color_{chain_id}_{res_num}, protein and chain {chain_id} and resi {res_num}\n")
        f.write('show_as("mesh"      ,"all")')
    print(f"PyMOL script '{output_script}' created successfully.")