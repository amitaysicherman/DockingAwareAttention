# Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration

**Official implementation of "Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration" (CIKM 2025).**


## Overview

Computational prediction of enzymatic reactions is a crucial challenge in sustainable chemical synthesis. Proteins, acting as catalysts, exhibit remarkable substrate adaptability, often catalyzing different transformations based on molecular partners. Current protein representation methods in reaction prediction often use static embeddings or ignore protein structure, failing to capture this dynamic adaptation.

Docking-Aware Attention (DAA) is a novel architecture that generates dynamic, context-dependent protein representations by incorporating molecular docking information into the attention mechanism. DAA combines physical interaction scores from docking predictions (e.g., using DiffDock) with learned attention patterns to focus on protein regions most relevant to specific molecular interactions. This approach better reflects the reality of enzyme behavior, where catalytic activity depends on specific substrate interactions.

Our method has demonstrated state-of-the-art performance on enzymatic reaction prediction and shows strong generalization to Drug-Target Interaction (DTI) prediction.

## Key Features & Contributions

* **Dynamic Protein Representations:** Generates protein embeddings that adapt to different molecular contexts.
* **Docking Integration:** Incorporates physical protein-ligand interaction scores (e.g., from DiffDock, TANKBind, or AutoDock) into the attention mechanism.
* **State-of-the-Art Performance:**
    * **Biocatalysis Prediction:** Achieves significant improvements, especially on complex molecules (9.5% relative improvement) and innovative reactions (12.3% relative improvement) on the ECREACT dataset.
    * **Drug-Target Interaction (DTI) Prediction:** Achieves state-of-the-art performance on the BioSNAP dataset (unseen protein split).
* **Interpretable Attention:** Produces attention patterns that adapt to different molecular contexts, offering insights into enzyme behavior.
* **General Framework:** Applicable to various protein-molecule interaction tasks beyond enzymatic synthesis planning.

## Installation

```bash
# Clone the repository
git clone [https://anonymous.4open.science/r/DockingAwareAttention-8B8E](https://anonymous.4open.science/r/DockingAwareAttention-8B8E) # Replace with actual GitHub link post-anonymization
cd DockingAwareAttention

# Create and activate conda environment
conda create -n daa python=3.8
conda activate daa

# Install dependencies
pip install -r requirements.txt
```
It is also necessary to install [DiffDock](https://github.com/gcorso/DiffDock) if you plan to use it for generating docking scores.

## Data Preparation

The model is trained on the ECREACT dataset, augmented with the USPTO dataset.

1.  **Download and Prepare ECREACT Dataset:**
    This script will download the ECREACT dataset and preprocess it into the required format.
    ```bash
    python preprocessing/prep_ecreact.py
    ```

2.  **Download and Prepare USPTO Dataset (for augmentation):**
    This script will download and preprocess the USPTO dataset.
    ```bash
    python preprocessing/prep_uspto.py
    ```

3.  **Generate Protein Embeddings (e.g., ESM3):**
    Protein embeddings are used as input to the DAA model. The paper uses ESM3-6B.
    This script requires an API token for the ESM Metagenomic Atlas API.
    ```bash
    # Get ESM3-6B embeddings (requires API token from Evolutionary Scale)
    python preprocessing/get_esm3_fold_emb.py --token YOUR_EVOLUTIONARY_SCALE_API_TOKEN
    ```
    Alternative embeddings can also be generated (optional, used in ablation studies):
    ```bash
    # ProtBERT embeddings
    python preprocessing/get_emb.py --model prot_bert

    # GearNet embeddings (requires precomputed PDB structures)
    python preprocessing/get_emb.py --model gearnet
    ```
    Ensure PDB files are available (e.g., from `get_esm3_fold_emb.py` or other sources) if using GearNet.

4.  **Prepare Molecular Docking Data:**
    DAA requires molecular docking scores. The primary method used in the paper is DiffDock.
    This script prepares the commands for running docking simulations (e.g., with DiffDock) for protein-molecule pairs from the ECREACT dataset.
    ```bash
    python preprocessing/prep_mol_prot_docking.py
    ```
    This will generate shell scripts (e.g., in `DiffDock/scripts/`) containing the commands to run. You will need to execute these scripts using your DiffDock installation.
    For example, if DiffDock is in a parallel directory:
    ```bash
    cd ../DiffDock
    bash scripts/run_1.sh # and so on for all generated scripts
    cd ../DockingAwareAttention
    ```
    After running docking, the interaction scores need to be processed if not already in the format expected by `dataset.py` (e.g. `get_reaction_attention_emd` in `preprocessing/docking_score.py` can generate EMD-based scores). The paper utilizes Lennard-Jones potentials averaged over multiple poses from DiffDock.

## Training

To train the DAA model for enzymatic reaction prediction:

```bash
python train.py \
    --ec_type 2 \
    --daa_type 4 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --epochs 10 \
    --emb_suf "" # Suffix for ESM3-6B embeddings, e.g., "_pb" for ProtBERT
```

**Key Arguments for `train.py`:**

* `--ec_type`: Enzyme classification type.
    * `0`: No EC information used.
    * `1`: Paper baseline (EC numbers as tokens).
    * `2`: Pretrained embeddings (DAA is typically used with this).
* `--daa_type`: Docking-Aware Attention mechanism type.
    * `0`: No DAA (standard attention with protein embedding).
    * `1`: Mean pooling of docking scores.
    * `2`: Docking scores used directly.
    * `3`: Attention mechanism only (no docking scores).
    * `4`: Full DAA (combining attention and docking scores, as per the paper).
* `--batch_size`: Training batch size.
* `--learning_rate`: Learning rate for the optimizer.
* `--epochs`: Number of training epochs.
* `--emb_suf`: Suffix for the protein embedding files (e.g., `""` for `embeddings.npy`, `"_pb"` for `embeddings_pb.npy`).
* `--add_ec_tokens`: (0 or 1) Whether to add EC tokens in addition to embeddings.
* `--emb_dropout`: Dropout rate for embeddings.
* `--concat_vec`: Concatenation strategy for protein embeddings (0, 1, or 2).

Refer to `train.py` for a full list of arguments.

## Evaluation

To evaluate a trained model on the test set:

```bash
python analyze_scores.py \
    --split test \
    --epoch 8.0 \
    --k 5 \
    --run_name "ec-ECType.PRETRAINED_daa-4_emb-0.0_ectokens-1" # Adjust to your run name
```
(Ensure the `run_name` corresponds to the output directory of your training run in `results/`)

This script will generate evaluation metrics, including:
* Top-k accuracy (for k=1, 3, 5 by default).
* Performance on complex molecules.
* Performance on innovative reactions.

## Results

### Biocatalysis Prediction (ECREACT Dataset)

Our model achieves state-of-the-art results on the ECREACT dataset:

| Method                      | All Top-1 (%) | All Top-3 (%) | All Top-5 (%) | Complex Top-1 (%) | Complex Top-3 (%) | Complex Top-5 (%) | Novel Top-1 (%) | Novel Top-3 (%) | Novel Top-5 (%) |
|-----------------------------|---------------|---------------|---------------|-------------------|-------------------|-------------------|-----------------|-----------------|-----------------|
| Chemical-Only \cite{kreutter2021predicting} & 37.29         | 54.65         | 61.39         | 30.78             | 44.92             | 51.56             | 18.25           | 33.75           | 40.82           |
| EC Tokens \cite{Probst2022}     | 46.01         | 61.71         | 66.64         | 34.26             | 51.22             | 56.79             | 31.42           | 44.46           | 49.45           |
| ESM3 \cite{hayes2024simulating}   | 44.58         | 60.64         | 65.62         | 34.23             | 48.57             | 54.11             | 27.39           | 42.35           | 47.47           |
| ProtBERT \cite{brandes2021proteinbert} | 43.49         | 59.52         | 64.82         | 33.11             | 45.48             | 50.42             | 31.21           | 40.12           | 44.66           |
| GearNet \cite{zhang2023protein}   | 43.95         | 60.10         | 63.84         | 28.09             | 40.80             | 46.23             | 20.16           | 32.62           | 39.13           |
| ReactEmbed\cite{sicherman2025reactembed} | 47.41         | 61.19         | 66.25         | 36.90             | 50.87             | 55.18             | 26.37           | 42.68           | 48.08           |
| READRetro \cite{kim2024readretro} | 46.71         | 61.45         | 66.44         | 35.58             | 51.04             | 55.98             | 28.89           | 43.57           | 48.76           |
| GSETransformer \cite{cong2025graph}| 40.39         | 57.08         | 63.15         | 31.94             | 45.2              | 50.99             | 24.73           | 36.93           | 42.74           |
| **DAA (Ours)** | **49.96** | **66.65** | **71.48** | **41.62** | **56.43** | **62.20** | **35.24** | **50.66** | **55.54** |

### Generalization to Drug-Target Interaction (DTI) Prediction

DAA-generated representations also achieve state-of-the-art performance on the BioSNAP DTI task (unseen protein split):

| Method                      | AUC          | Sensitivity  | Specificity  |
|-----------------------------|--------------|--------------|--------------|
| Top-DTI    | 75.01        | 69.31        | 68.67        |
| **Our Model (DAA-based)** | **76.12** | **71.45** | **69.21** |
*(Refer to the paper for the full list of DTI baselines)*

## Citation

If you find this code or our work useful for your research, please cite our paper:

```bibtex
@inproceedings{sicherman2025daa,
  title={Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration},
  author={Sicherman, Amitay and Radinsky, Kira},
  booktitle={Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2025},
  month={November},
  address={Seoul, Korea},
  doi={TBD.TBD},
  isbn={978-1-4503-XXXX-X/18/06}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

We thank the authors of ESM, DiffDock, and ECREACT for making their code and data publicly available.

## GenAI Usage Disclosure

The paper's text was partially revised for clarity and form with the assistance of Google's Gemini 2.5 Pro. No generative AI was used for this work's data, code, or experimental design aspects.
