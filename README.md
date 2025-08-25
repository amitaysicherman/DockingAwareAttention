# Docking-Aware Attention (DAA)

**Official implementation of "Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration".**

[![Paper](https://img.shields.io/badge/paper-CIKM'25-B31B1B.svg)](https://link_to_your_paper.com) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Computational prediction of enzymatic reactions is a crucial challenge in sustainable chemistry. Proteins, as catalysts, are highly adaptable and often perform different transformations based on their molecular partners. However, current protein representation methods often fail to capture this dynamic adaptation, using static embeddings that are blind to the specific substrate.

**Docking-Aware Attention (DAA)** is a novel architecture that generates **dynamic, context-dependent protein representations**. By integrating physical interaction scores from molecular docking (e.g., using DiffDock) directly into an attention mechanism, DAA learns to focus on protein regions most relevant to a *specific* molecular interaction. This approach better reflects the dynamic reality of enzyme behavior and leads to state-of-the-art performance in enzymatic reaction prediction.

<br>

![DAA Architecture Diagram](https://github.com/amitaysicherman/DockingAwareAttention/blob/main/figures/daa_update.png?raw=true)
*Figure: The DAA architecture generates docking scores (left stream) which are used to bias a self-attention mechanism operating on protein sequence embeddings (right stream), producing a final context-aware representation.*

---

## Key Features

* **Dynamic Protein Representations**: Generates protein embeddings that adapt to different molecular contexts.
* **Physics-Informed AI**: Integrates physical protein-ligand interaction scores (from docking tools like DiffDock, TANKBind, etc.) into the attention mechanism.
* **State-of-the-Art Performance**:
    * **Biocatalysis**: Achieves a **9.5%** relative improvement on complex molecules and a **12.3%** relative improvement on innovative reactions on the ECREACT dataset.
    * **Drug-Target Interaction (DTI)**: Achieves state-of-the-art performance on the BioSNAP dataset (unseen protein split), demonstrating strong generalization.
* **Interpretable Attention**: Produces attention patterns that highlight key interaction regions, offering insights into enzyme behavior.
* **General Framework**: Applicable to various protein-molecule interaction tasks beyond enzymatic synthesis.

---

## Getting Started

### 1. Installation

First, clone the repository and set up the conda environment.

```bash
# Clone the repository
git clone [https://github.com/amitaysicherman/DockingAwareAttention](https://github.com/amitaysicherman/DockingAwareAttention)
cd DockingAwareAttention

# Create and activate a conda environment
conda create -n daa python=3.8
conda activate daa

# Install dependencies
# It is recommended to install rdkit via conda first for stability
conda install -c conda-forge rdkit
pip install -r requirements.txt
````

Our framework uses docking scores to inform the attention mechanism. If you plan to generate these scores yourself, you will also need to install a docking tool. The primary tool used in the paper is **[DiffDock](https://github.com/gcorso/DiffDock)**.

### 2\. Data Preparation

The model is trained on the ECREACT dataset, augmented with USPTO. The following scripts will download and preprocess the necessary data.

```bash
# 1. Download and preprocess the ECREACT dataset
python prep_ecreact.py 

# 2. Download and preprocess the USPTO dataset for augmentation
python prep_uspto.py

# 3. Fetch Protein IDs and FASTA sequences from UniProt
# This script uses the EC numbers from ECREACT to get corresponding protein data.
python prep_proteins.py

# 4. Generate Protein Embeddings (e.g., from ESM)
# The paper uses ESM3-6B embeddings. This script generates them.
# Other embeddings can be used as detailed in the paper's ablation studies.
python get_esm_fold_emb.py 

# 5. Prepare and run Molecular Docking
# This script generates shell scripts to run docking for all protein-molecule pairs.
python prep_mol_prot_docking.py

# Execute the generated docking scripts using your DiffDock installation.
# For example, if DiffDock is in a parallel directory:
cd ../DiffDock
bash scripts/run_1.sh
cd ../DockingAwareAttention

# After docking, process the results to get interaction scores.
# The function `get_reaction_attention_emd` in `preprocessing/docking_score.py` can be used for this.
```

-----

## Model Training

To train the full DAA model for enzymatic reaction prediction, run the following command. The arguments correspond to the best-performing model in the paper.

```bash
python train.py \
    --ec_type 2 \
    --daa_type 4 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --epochs 10 \
    --emb_suf ""
```

### Key Training Arguments

  * `--ec_type`: Specifies how protein information is used.
      * `0`: No EC information.
      * `1`: EC numbers as simple tokens (paper baseline).
      * `2`: **Use pre-trained protein embeddings** (required for DAA).
  * `--daa_type`: Specifies the Docking-Aware Attention mechanism.
      * `0`: No DAA (standard attention with a static protein embedding).
      * `1`: Mean pooling of docking scores.
      * `2`: Docking scores used directly to weight embeddings.
      * `3`: Attention mechanism only (no docking scores).
      * `4`: **Full DAA**, combining learned attention and docking scores (as per the paper).
  * `--emb_suf`: Suffix for the protein embedding files (e.g., `""` for `embeddings.npy`, `"_pb"` for ProtBERT `embeddings_pb.npy`).
  * `--add_ec_tokens`: (0 or 1) Whether to add EC tokens in addition to protein embeddings.
  * `--concat_vec`: Strategy for incorporating the protein vector (`0`: prepend as new token, `1`: add to all tokens, `2`: concat to all tokens).

-----

## Evaluation

To evaluate a trained model on the test set, use the `analyze_scores.py` script. You'll need to specify the run name, which is generated automatically during training based on the hyperparameters.

```bash
# Example evaluation command for a trained model
python analyze_scores.py \
    --split test \
    --epoch 9.0 \
    --k 5 \
    --run_name "ec-ECType.PRETRAINED_daa-4_emb-0.0_ectokens-0"
```

  * Make sure the `--run_name` corresponds to the output directory of your training run in the `results/` folder.
  * This script will report Top-k accuracy and performance on the "Complex" and "Innovative" subsets.

-----

## Results

### Biocatalysis Prediction (ECREACT Dataset)

Our model achieves state-of-the-art results, showing significant improvements over previous methods, especially on challenging subsets.

| Method                      | All Top-5 (%) | Complex Top-5 (%) | Novel Top-5 (%) |
|-----------------------------|---------------|-------------------|-----------------|
| EC Tokens      | 66.64         | 56.79             | 49.45           |
| ESM3    | 65.62         | 54.11             | 47.47           |
| READRetro | 66.44         | 55.98             | 48.76           |
| **DAA (Ours)** | **71.48** | **62.20** | **55.54** |

### Generalization to Drug-Target Interaction (DTI)

DAA-generated representations also achieve state-of-the-art performance on the BioSNAP DTI task (unseen protein split), demonstrating excellent generalization.

| Method                      | AUC          |
|-----------------------------|--------------|
| MolTrans                    | 71.42        |
| DrugLAMP                    | 73.23        |
| Top-DTI                     | 75.01        |
| **Our Model (DAA-based)** | **76.12** |

-----

## How to Cite

If you use our work, please cite the following paper:

```bibtex
@inproceedings{sicherman2025daa,
  title={Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration},
  author={Sicherman, Amitay and Radinsky, Kira},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025},
  series = {CIKM '25}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

We thank the authors of ESM, DiffDock, and ECREACT for making their code and data publicly available.
