import os


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
        return self.ec_to_id[ec]

    def get_base_dir(self, id_):
        if id_ not in self.id_to_chunk:
            return None
        chunk = self.get_chunk(id_)
        return f"datasets/ecreact/proteins/chunk_{chunk}/{id_}"

    def get_pdb_file(self, id_):
        base_dir = self.get_base_dir(id_)
        if not base_dir:
            return None
        return f"{base_dir}/fold.pdb"


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
