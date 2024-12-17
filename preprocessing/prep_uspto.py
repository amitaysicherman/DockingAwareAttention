import os
from rdkit import Chem
from preprocessing.tokenizer_utils import tokenize_reaction_smiles
from tqdm import tqdm
from rdkit import rdBase

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.warning')




output_dir = "datasets/uspto"
os.makedirs(output_dir, exist_ok=True)

src_test = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-test.txt"
src_train_1 = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-train-split-1.txt"
src_train_2 = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-train-split-2.txt"
src_valid = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/src-valid.txt"
tgt_test = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-test.txt"
tgt_train = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-train.txt"
tgt_valid = "https://raw.githubusercontent.com/rxn4chemistry/OpenNMT-py/carbohydrate_transformer/data/uspto_dataset/tgt-valid.txt"

for url in [src_test, src_train_1, src_train_2, src_valid, tgt_test, tgt_train, tgt_valid]:
    file_path = os.path.join(output_dir, os.path.basename(url))
    if not os.path.isfile(file_path):
        os.system(f"curl -o {file_path} {url}")

# combine train splits
lines = []
src_split_1 = os.path.join(output_dir, 'src-train-split-1.txt')
src_split_2 = os.path.join(output_dir, 'src-train-split-2.txt')
output_file = os.path.join(output_dir, 'src-train.txt')
with open(src_split_1) as f:
    lines.extend(f.readlines())
with open(src_split_2) as f:
    lines.extend(f.readlines())
with open(output_file, 'w') as f:
    f.writelines(lines)

os.remove(src_split_1)
os.remove(src_split_2)


def remove_mol_stereochemistry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    no_stereo_smiles = Chem.MolToSmiles(mol, canonical=True)
    return no_stereo_smiles


def remove_stereochemistry(mols_smiles):
    mols_smiles = [remove_mol_stereochemistry(m) for m in mols_smiles.split('.')]
    if None in mols_smiles:
        return None
    mols_smiles = '.'.join(mols_smiles)
    return mols_smiles


for split in ["train", "valid", "test"]:
    src_file = os.path.join(output_dir, f"src-{split}.txt")
    tgt_file = os.path.join(output_dir, f"tgt-{split}.txt")
    with open(src_file) as f:
        src_lines = f.read().splitlines()
    with open(tgt_file) as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines)
    updated_src_lines = []
    updated_tgt_lines = []
    for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        src_line = remove_stereochemistry(src_line.replace(" ", ""))
        tgt_line = remove_stereochemistry(tgt_line.replace(" ", ""))
        if src_line is not None and tgt_line is not None:
            updated_src_lines.append(tokenize_reaction_smiles(src_line))
            updated_tgt_lines.append(tokenize_reaction_smiles(tgt_line))
    with open(src_file, "w") as f:
        f.write("\n".join(updated_src_lines))
    with open(tgt_file, "w") as f:
        f.write("\n".join(updated_tgt_lines))
