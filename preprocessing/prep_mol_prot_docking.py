from utils import ProteinsManager, MoleculeManager
import os


def line_to_ec_mols(line):
    src, ec = line.split('>>')[0].split('|')
    mols = src.split('.')
    return mols, ec


protein_manager = ProteinsManager()
molecule_manager = MoleculeManager()
seen_pairs = set()
with open("datasets/ecreact/ecreact-1.0.txt", "r") as f:
    lines = f.read().splitlines()
base_cmd = "python -m inference --config default_inference_args.yaml"
cmds = []
for line in lines:
    mols, ec = line_to_ec_mols(line)
    protein_id = protein_manager.get_id(ec)
    pdb_file = protein_manager.get_pdb_file(protein_id)
    protein_base_dir = protein_manager.get_base_dir(protein_id)
    if not pdb_file:
        print(f"Missing PDB file for {ec}")
        continue
    if not os.path.exists(pdb_file):
        print()
        print(f"PDB file {pdb_file} does not exist")
        continue
    for mol in mols:
        mol_id = molecule_manager.get_id(mol)
        if not mol_id:
            print(f"Missing mol ID for {mol}")
            continue
        if (protein_id, mol_id) in seen_pairs:
            continue
        seen_pairs.add((protein_id, mol_id))
        output_dir = f'../{protein_base_dir}/{mol_id}'  # run from DiffDock directory, so need to go up one level
        cmd = f"{base_cmd} --protein_path '../{pdb_file}' --ligand '{mol}' --out_dir '{output_dir}'"
        cmds.append(cmd)
print(f"{len(cmds)} docking runs")
scripts_dir = "DiffDock/scripts"
os.makedirs(scripts_dir, exist_ok=True)

n_splits = 200
for i in range(n_splits):
    with open(f"{scripts_dir}/run_{i + 1}.sh", 'w') as f:
        for cmd in cmds[i::n_splits]:
            f.write(f"{cmd}\n")
