import os
import pandas as pd
import numpy as np
from preprocessing.tokenizer_utils import tokenize_enzymatic_reaction_smiles

dir_path = os.path.join('datasets', 'ecreact')
os.makedirs(dir_path, exist_ok=True)

url = "https://raw.githubusercontent.com/rxn4chemistry/biocatalysis-model/main/data/ecreact-1.0.csv"
file_path = os.path.join(dir_path, 'ecreact-1.0.csv')
if not os.path.isfile(file_path):
    os.system(f"curl -o {file_path} {url}")
os.path.isfile(file_path), file_path

output_file = os.path.join(dir_path, 'ecreact-1.0.txt')
df = pd.read_csv(file_path)
txt_file = []
for i in range(len(df)):
    txt_file.append(df['rxn_smiles'][i])
with open(output_file, 'w') as f:
    for item in txt_file:
        # item = remove_stereochemistry(item)
        f.write("%s\n" % item)

output_tokenized_file = os.path.join(dir_path, 'ecreact-1.0-tokenized.txt')
tokens_lines = []
src = []
tgt = []
datasets = []
all_ec = set()

ecreact = pd.read_csv(file_path)
with open(output_tokenized_file, 'w') as f2:
    for i, row in ecreact.iterrows():
        rnx = row['rxn_smiles']
        source = row['source']
        all_ec.add(row['ec'])
        tokens = tokenize_enzymatic_reaction_smiles(rnx)
        if tokens:
            src_, tgt_ = tokens.split(' >> ')
            src.append(src_)
            tgt.append(tgt_)
            datasets.append(source)
            f2.write(tokens + '\n')

ec_file = os.path.join(dir_path, 'ec.txt')
with open(ec_file, 'w') as f:
    for item in all_ec:
        f.write("%s\n" % item)
print(f"Total EC numbers: {len(all_ec)}")


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