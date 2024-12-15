import os
import pandas as pd
import numpy as np
from preprocessing.tokenizer_utils import tokenize_enzymatic_reaction_smiles
from rdkit import Chem

from rdkit import rdBase

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.warning')


def remove_mol_stereochemistry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    no_stereo_smiles = Chem.MolToSmiles(mol, canonical=True)
    return no_stereo_smiles


def remove_stereochemistry(rxn_smiles):
    src_ec, tgt = rxn_smiles.split('>>')
    src, ec = src_ec.split('|')
    src = ".".join([remove_mol_stereochemistry(m) for m in src.split('.')])
    tgt = ".".join([remove_mol_stereochemistry(m) for m in tgt.split('.')])
    return f"{src}|{ec}>>{tgt}"


dir_path = os.path.join('datasets', 'ecreact')
os.makedirs(dir_path, exist_ok=True)

url = "https://raw.githubusercontent.com/rxn4chemistry/biocatalysis-model/main/data/ecreact-1.0.csv"
file_path = os.path.join(dir_path, 'ecreact-1.0.csv')
if not os.path.isfile(file_path):
    os.system(f"curl -o {file_path} {url}")
os.path.isfile(file_path), file_path

output_tokenized_file = os.path.join(dir_path, 'ecreact-1.0-tokenized.txt')
text_file = os.path.join(dir_path, 'ecreact-1.0.txt')

tokens_lines = []
text_lines = []
src = []
tgt = []
datasets = []
all_ec = set()
all_src_molecules = set()

df = pd.read_csv(file_path)
for index, row in df.iterrows():
    rnx = remove_stereochemistry(row['rxn_smiles'])
    source = row['source']
    all_ec.add(row['ec'])
    tokens = tokenize_enzymatic_reaction_smiles(rnx)
    if tokens:
        text_lines.append(rnx)
        molecules = rnx.split('>>')[0].split('|')[0].split('.')
        all_src_molecules.update(molecules)
        src_, tgt_ = tokens.split(' >> ')
        src.append(src_)
        tgt.append(tgt_)
        datasets.append(source)
        tokens_lines.append(tokens)

with open(output_tokenized_file, 'w') as f:
    for line in tokens_lines:
        f.write(line + '\n')
with open(text_file, 'w') as f:
    for line in text_lines:
        f.write(line + '\n')

ec_file = os.path.join(dir_path, 'ec.txt')
with open(ec_file, 'w') as f:
    for item in all_ec:
        f.write("%s\n" % item)
print(f"Total EC numbers: {len(all_ec)}")

molecules_file = os.path.join(dir_path, 'molecules.txt')
with open(molecules_file, 'w') as f:
    for item in all_src_molecules:
        f.write("%s\n" % item)
print(f"Total molecules: {len(all_src_molecules)}")

print(f"Tokenized {len(src)} reactions")
assert len(src) == len(tgt)
output_dir_files = dir_path
os.makedirs(output_dir_files, exist_ok=True)
# split into train val test 70 15 15
np.random.seed(42)
indices = np.random.permutation(len(src))
train_indices = indices[:int(0.7 * len(src))]
val_indices = indices[int(0.7 * len(src)):int(0.85 * len(src))]
test_indices = indices[int(0.85 * len(src)):]
for split, split_indices in zip(['train', 'valid', 'test'], [train_indices, val_indices, test_indices]):
    split_src = [src[i] for i in split_indices]
    split_tgt = [tgt[i] for i in split_indices]
    split_dst = [datasets[i] for i in split_indices]
    print(f"{split}: {len(split_src)}")
    with open(os.path.join(output_dir_files, f'src-{split}.txt'), 'w') as f:
        for line in split_src:
            f.write(f"{line}\n")
    with open(os.path.join(output_dir_files, f'tgt-{split}.txt'), 'w') as f:
        for line in split_tgt:
            f.write(f"{line}\n")
    with open(os.path.join(output_dir_files, f'datasets-{split}.txt'), 'w') as f:
        for line in split_dst:
            f.write(f"{line}\n")
