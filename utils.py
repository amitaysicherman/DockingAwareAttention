import os
from enum import Enum

from rdkit import Chem
# Suppress RDKit warnings
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')


class ProteinsManager:
    def __init__(self):
        self.id_to_chunk = {}
        with open("datasets/ecreact/proteins/id_to_chunk.txt", "r") as f:
            for line in f:
                id_, chunk = line.strip().split()
                self.id_to_chunk[id_] = int(chunk)
        self.ec_to_id = {}
        self.id_to_ec = {}
        with open("datasets/ecreact/ec.txt", "r") as f:
            ec_lines = f.read().splitlines()
        with open("datasets/ecreact/ec_ids.txt", "r") as f:
            id_lines = f.read().splitlines()
        assert len(ec_lines) == len(id_lines)
        for ec, id_ in zip(ec_lines, id_lines):
            self.ec_to_id[ec] = id_
            self.id_to_ec[id_] = ec

    def get_chunk(self, id_):
        return self.id_to_chunk[id_]

    def get_ec(self, id_):
        return self.id_to_ec[id_]

    def get_id(self, ec):
        if "[" in ec:  # tokenized
            ec = ec.replace(" ", "")
            ec = ec.replace("v", "").replace("u", "").replace("t", "").replace("q", "").replace("[", "")
            ec = ".".join(ec.split("]")[:-1])
        if ec not in self.ec_to_id:
            return None

        return self.ec_to_id[ec]

    def get_base_dir(self, id_):
        if id_ not in self.id_to_chunk:
            return None
        chunk = self.get_chunk(id_)
        base_dir = f"datasets/ecreact/proteins/chunk_{chunk}/{id_}"
        if not os.path.exists(base_dir):
            return None
        return base_dir

    def get_pdb_file(self, id_):
        base_dir = self.get_base_dir(id_)
        if not base_dir:
            return None
        pdb_file = f"{base_dir}/fold.pdb"
        if not os.path.exists(pdb_file):
            return None
        return pdb_file

    def get_emb_file(self, id_, suf=""):
        base_dir = self.get_base_dir(id_)
        if not base_dir:
            return None
        emb_file = f"{base_dir}/embeddings{suf}.npy"
        if not os.path.exists(emb_file):
            return None
        return emb_file


class MoleculeManager:
    def __init__(self):
        self.mol_to_id = {}
        self.id_to_mol = {}
        with open("datasets/ecreact/molecules.txt", "r") as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                self.mol_to_id[line] = i
                self.id_to_mol[i] = line

    def get_id(self, mol):
        if " " in mol:  # tokenized
            mol = mol.replace(" ", "")
        if mol not in self.mol_to_id:
            return None
        return self.mol_to_id[mol]

    def get_mol(self, id_):
        if id_ not in self.id_to_mol:
            return None
        return self.id_to_mol[id_]

    def get_id_token_mol(self, mol):
        mol = mol.replace(' ', '')
        return self.mol_to_id[mol], mol


def get_prot_mol_doc_file(protein_base, molecule_id):
    if not os.path.exists(protein_base):
        return None
    dock_dir = f'{protein_base}/{molecule_id}'
    if not os.path.exists(dock_dir) or not os.path.exists(f'{dock_dir}/complex_0'):
        return None
    dock_file = f'{protein_base}/{molecule_id}/complex_0/rank1.sdf'
    if not os.path.exists(dock_file):
        return None
    return dock_file


class DaaType(Enum):
    NO = 0
    MEAN = 1
    DOCKING = 2
    ATTENTION = 3
    ALL = 4


class ECType(Enum):
    NO_EC = 0
    PAPER = 1
    PRETRAINED = 2


def tokens_to_canonical_smiles(tokenizer, tokens, remove_stereo=False):
    smiles = tokenizer.decode(tokens, skip_special_tokens=True)
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)
