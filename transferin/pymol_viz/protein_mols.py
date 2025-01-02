import numpy as np
import matplotlib.pyplot as plt
from transferin.pymol_viz.utils import get_residue_ids_from_pdb
from sklearn.preprocessing import MinMaxScaler

v_cmap = plt.get_cmap("Greens")
TAB10_COLORS = plt.get_cmap("tab10").colors


def create_pymol_script_with_sdf(pdb_file: str, sdf_files: list, color_values,
                                 output_script: str = "protein_molecules_colored.pml"):
    residue_ids = get_residue_ids_from_pdb(pdb_file)
    assert len(residue_ids) == len(color_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_pca_values = scaler.fit_transform(color_values.reshape(-1, 1)).flatten()
    viridis_colors = plt.cm.get_cmap("Greens")
    emb_colors = viridis_colors(normalized_pca_values)
    emb_colors = emb_colors[:, :3]

    with open(output_script, 'w') as f:
        f.write(f"load {pdb_file}, protein\n")
        for i, sdf_file in enumerate(sdf_files):
            molecule_name = f"molecule_{i + 1}"
            f.write(f"load {sdf_file}, {molecule_name}\n")
            f.write(f"show sticks, {molecule_name}\n")
            f.write(f"color red,{molecule_name} \n")
        for (chain_id, res_num), value in zip(residue_ids, emb_colors):
            r, g, b = value
            f.write(f"set_color color_{chain_id}_{res_num}, [{r}, {g}, {b}]\n")
            f.write(f"color color_{chain_id}_{res_num}, protein and chain {chain_id} and resi {res_num}\n")
        f.write("show cartoon, protein\n")
    print(f"PyMOL script '{output_script}' created successfully.")
