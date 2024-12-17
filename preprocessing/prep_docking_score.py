import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import re
from rdkit import Chem
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from tqdm import tqdm
from utils import ProteinsManager, MoleculeManager, get_prot_mol_doc_file

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")
aa3to1 = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
    'MSE': 'M',
}
CA_PAT = re.compile(r"^ATOM\s+\d+\s+CA\s+([A-Z]{3})\s+([\w])\s+\d+\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)")


def get_protein_cords(pdb_file):
    seq = []
    cords = []
    with open(pdb_file, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith("ENDMDL"):
                break
            match = CA_PAT.match(line)
            if match:
                resn = match.group(1)  # Residue name (e.g., SER)
                chain = match.group(2)  # Chain identifier (e.g., A)
                assert chain == "A"  # For esmfold from fasta
                x_coord = float(match.group(3))  # X coordinate
                y_coord = float(match.group(4))  # Y coordinate
                z_coord = float(match.group(5))  # Z coordinate
                seq.append(aa3to1.get(resn, 'X'))
                cords.append([x_coord, y_coord, z_coord])
    return "".join(seq), cords


def get_mol_cords(sdf_file):
    coords = []
    supplier = Chem.SDMolSupplier(sdf_file)
    for mol in supplier:
        if mol is None:
            continue
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
    return coords


def calculate_dsw(distances, vdw_products=1.7, clip_value=1.91):
    distances = np.clip(distances, clip_value, None)
    return (1 / distances) * (2 * np.power(vdw_products / distances, 12) - np.power(vdw_products / distances, 6))


def get_protein_mol_att(protein_manager: ProteinsManager, protein_id, molecule_id):
    protein_file = protein_manager.get_pdb_file(protein_id)
    protein_seq, protein_cords = get_protein_cords(protein_file)
    protein_cords = np.array(protein_cords)
    sdf_file = get_prot_mol_doc_file(protein_manager.get_base_dir(protein_id), molecule_id)
    if sdf_file is None:
        return None
    lig_coords = get_mol_cords(sdf_file)
    if len(lig_coords) == 0:
        return None
    dist = euclidean_distances(protein_cords, lig_coords)
    weights = calculate_dsw(dist)
    weights = weights.mean(axis=1)
    return weights


def get_reaction_attention_emd(rnx, protein_manager: ProteinsManager, molecule_manager: MoleculeManager, tokens=False,
                               only_src=False):
    if tokens:
        rnx = rnx.replace(" ", "")
    if not only_src:
        rnx = rnx.split(">>")[0]
    src, ec = rnx.split("|")
    protein_id = protein_manager.get_id(ec)
    if protein_id is None:
        return None
    weights = []
    for mol_smiles in src.split("."):
        mol_id = molecule_manager.get_id(mol_smiles)
        if mol_id is None:
            continue
        w = get_protein_mol_att(protein_manager, protein_id, mol_id)
        if w is not None:
            weights.append(w)
    if len(weights) == 0:
        return None
    weights = np.array(weights).mean(axis=0)  # Average over all molecules
    weights = np.concatenate([[0], weights, [0]])  # Add start and end tokens
    return weights


if __name__ == "__main__":

    protein_manager = ProteinsManager()
    molecule_manager = MoleculeManager()
    for split in ["train", "valid", "test"]:
        split_w = []
        with open(f"datasets/ecreact/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        for line in tqdm(src_lines):
            w = get_reaction_attention_emd(line, protein_manager, molecule_manager, tokens=True, only_src=True)
            if w is None:
                w = ""
            else:
                w = " ".join([str(x) for x in w])
            split_w.append(w)
        with open(f"datasets/ecreact/w_{split}.txt", "w") as f:
            f.write("\n".join(split_w))
