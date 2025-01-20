# Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration

Official implementation of "Docking-Aware Attention: Dynamic Protein Representations through Molecular Context Integration" (KDD 2025).

## Overview

Docking-Aware Attention (DAA) is a novel architecture that generates dynamic, context-dependent protein representations by incorporating molecular docking information into the attention mechanism. This repository contains the implementation of DAA and the code to reproduce the results from our paper.

## Key Features

- Dynamic protein representations that adapt to different molecular contexts
- Integration of physical interaction scores from ensemble docking predictions
- State-of-the-art performance on enzymatic reaction prediction
- Interpretable attention patterns that adapt to different molecular contexts

## Installation

```bash
# Clone the repository
git clone https://github.com/XXX/docking-aware-attention
cd docking-aware-attention

# Create and activate conda environment
conda create -n daa python=3.8
conda activate daa

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

1. Download and prepare the ECREACT dataset:
```bash
python preprocessing/prep_ecreact.py
```

2. Generate protein embeddings:
```bash
# Get ESM-3 embeddings (requires Anthropic API token)
python preprocessing/get_esm3_fold_emb.py --token YOUR_TOKEN

# Alternative embeddings (optional)
python preprocessing/get_emb.py --model prot_bert
python preprocessing/get_emb.py --model gearnet
```

3. Prepare molecular docking data:
```bash
python preprocessing/prep_mol_prot_docking.py
```

## Training

To train the DAA model:

```bash
python train.py \
    --ec_type 2 \
    --daa_type 4 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --epochs 10
```

Key arguments:
- `ec_type`: Enzyme classification type (0: No EC, 1: Paper baseline, 2: Pretrained)
- `daa_type`: Attention mechanism (0: No DAA, 1: Mean, 2: Docking, 3: Attention, 4: Full DAA)
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `epochs`: Number of training epochs

## Evaluation

To evaluate a trained model:

```bash
python analyze_scores.py \
    --split test \
    --epoch 8.0 \
    --k 5
```

This will generate evaluation metrics including:
- Top-k accuracy (k=1,3,5)
- Performance on complex molecules
- Performance on innovative reactions


## Results

Our model achieves state-of-the-art results on the ECREACT dataset:

| Method | Top-1 | Top-3 | Top-5 |
|--------|-------|-------|-------|
| Baseline | 37.29 | 54.65 | 61.39 |
| EC Tokens | 47.01 | 61.71 | 66.64 |
| DAA (Ours) | **49.96** | **65.65** | **70.48** |

## Citation

If you find this code useful for your research, please cite our paper:


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the authors of ESM, DiffDock, and ECREACT for making their code and data publicly available.