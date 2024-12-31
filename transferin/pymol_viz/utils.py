from collections import defaultdict

import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
import numpy as np
from tqdm import tqdm

def get_residue_ids_from_pdb(pdb_file):
    """
    Extracts residue IDs from a PDB file in the order they appear in the file.
    Returns a list of residue IDs that correspond to the protein sequence.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # We use the residue ID as a tuple of (chain ID, residue sequence number, insertion code)
                residue_id = (chain.id, residue.id[1])
                residue_ids.append(residue_id)
    return residue_ids


def remove_stereo(s):
    s = s.replace(" ", "")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def load_maps():
    with open("datasets/docking/smiles_to_id.txt") as f:
        id_to_smile = {int(x.split()[1]): x.split()[0] for x in f.readlines()}
        smile_to_id = {x.split()[0]: int(x.split()[1]) for x in f.readlines()}
    ec_mapping = pd.read_csv("datasets/ec_map.csv")
    uniport_to_ec = defaultdict(str)
    for i, row in ec_mapping.iterrows():
        uniport_to_ec[row["Uniprot_id"]] = row["EC_full"]
    return id_to_smile, smile_to_id, uniport_to_ec


def remove_dup_mis_mols(molecules_ids, id_to_smile):
    molecules_smiles = [id_to_smile[int(x)] for x in molecules_ids]
    molecules_smiles_no_stereo = [remove_stereo(x) for x in molecules_smiles]
    molecules_smiles_mask = [True] * len(molecules_smiles)
    for i in range(1, len(molecules_smiles)):
        if molecules_smiles_no_stereo[i] is None:
            molecules_smiles_mask[i] = False
        if molecules_smiles_no_stereo[i] in molecules_smiles_no_stereo[:i]:
            molecules_smiles_mask[i] = False
    return [molecules_ids[i] for i in range(len(molecules_ids)) if molecules_smiles_mask[i]]


def load_molecules(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    molecules = [mol for mol in supplier if mol is not None]
    return molecules


def calculate_average_distance(mol1, mol2):
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        raise ValueError("The molecules must have the same number of atoms.")
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    from sklearn.metrics.pairwise import euclidean_distances
    all_dist=euclidean_distances(conf1.GetPositions(), conf2.GetPositions())
    avg_distance = np.mean(all_dist)
    return avg_distance

def filter_molecule_by_len(mols_files, min_len_ratio):
    mols = []
    files_with_mol = []
    for file in mols_files:
        new_mols = load_molecules(file)
        if len(new_mols) == 0:
            continue
        if len(new_mols) > 1:
            continue
        files_with_mol.append(file)
        mols.append(new_mols[0])
    conf = mols[0].GetConformer()
    mol_xyz = [conf.GetAtomPosition(i) for i in range(mols[0].GetNumAtoms())]
    max_size = max(
        [np.linalg.norm(mol_xyz[i] - mol_xyz[j]) for i in range(len(mol_xyz)) for j in range(i + 1, len(mol_xyz))])
    min_len = min_len_ratio * max_size
    pair_dist = np.zeros((len(mols), len(mols)))
    for i in range(len(mols)):
        for j in range(0,i + 1):
            pair_dist[i, j] = calculate_average_distance(mols[i], mols[j])
    mols_files_filtered = [files_with_mol[0]]
    for i in range(1, len(mols)):
        if np.all(pair_dist[i, :i] > min_len):
            mols_files_filtered.append(files_with_mol[i])
    return mols_files_filtered


if __name__ == "__main__":
    from glob import glob

    mols_file = glob("/Users/amitay.s/PycharmProjects/BioCatalystChem/datasets/docking2/A0A009HWM5/0/complex_0/*.sdf")
    filter_molecule_by_len(mols_file, 0.5)