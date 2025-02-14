import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import re
from rdkit import Chem
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from tqdm import tqdm
from utils import ProteinsManager, MoleculeManager, get_prot_mol_doc_file
EPS = 1e-6
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


def get_protein_mol_att(protein_manager: ProteinsManager, protein_id, molecules_id):
    protein_file = protein_manager.get_pdb_file(protein_id)
    if protein_file is None:
        return None
    try:
        protein_seq, protein_cords = get_protein_cords(protein_file)
    except Exception as e:
        return None
    protein_cords = np.array(protein_cords)
    all_mol_coords = []
    for mol_id in molecules_id:
        sdf_file = get_prot_mol_doc_file(protein_manager.get_base_dir(protein_id), mol_id)
        if sdf_file is None:
            continue
        try:
            mol_cords = get_mol_cords(sdf_file)
        except Exception as e:
            continue
        if len(mol_cords) == 0:
            continue
        all_mol_coords.extend(mol_cords)
    if len(all_mol_coords) == 0:
        return None
    all_mol_coords = np.array(all_mol_coords)
    dist = euclidean_distances(protein_cords, all_mol_coords)
    weights = calculate_dsw(dist)
    weights = weights.mean(axis=1)
    weights = weights / (weights.sum() + EPS)
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
    mol_ids = [molecule_manager.get_id(mol_smiles) for mol_smiles in src.split(".")]
    mol_ids = [mol_id for mol_id in mol_ids if mol_id is not None]
    weights = get_protein_mol_att(protein_manager, protein_id, mol_ids)
    if weights is None:
        return None
    weights = np.concatenate([[0], weights, [0]])  # Add start and end tokens
    return weights


if __name__ == "__main__":

    protein_manager = ProteinsManager()
    molecule_manager = MoleculeManager()
    for split in ["train", "valid", "test"]:
        split_w = []
        with open(f"datasets/ecreact/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        pbar = tqdm(src_lines)
        failed = 0
        total = 0
        for line in pbar:
            w = get_reaction_attention_emd(line, protein_manager, molecule_manager, tokens=True, only_src=True)
            if w is None:
                w = ""
                failed += 1
            else:
                w = " ".join([str(x) for x in w])
            split_w.append(w)
            total += 1
            msg = f"Failed: {failed}/{total}"
            pbar.set_postfix_str(msg)

        with open(f"datasets/ecreact/w_{split}.txt", "w") as f:
            f.write("\n".join(split_w))
