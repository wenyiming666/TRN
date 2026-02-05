# Overcoming Shortcut Learning in RNA-Small Molecule Modeling via Bias-Matched Decoys and Structure-Aware Network Design

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/dataset-RNAdecoyDB-yellow)](data/)


## 📌 Abstract

Targeting RNA with small molecules represents a promising frontier in drug discovery, yet current computational methods are hindered by the limited quality and intrinsic bias of available training data. In this study, we reveal a critical deficiency in widely used benchmark datasets (e.g., ROBIN), in which negative samples are predominantly separated from positive RNA-ligand pairs by trivial physicochemical property differences rather than true structural incompatibility. Such biases enable **shortcut learning**, leading to artificially inflated predictive performance that fails to reflect molecular recognition.

To overcome this limitation, we introduce:
1.  **RNAdecoyDB**: A rigorously constructed hard-negative dataset generated through a feature-matching strategy that aligns the physicochemical property distributions between positive and negative samples.
2.  **Target RNA Network (TRN)**: A robust predictive model designed to capture intrinsic RNA-ligand structural interactions.
3.  **Zinc-TRN**: A library of **6.96 million** putative RNA-binding molecules screened from the ZINC database, providing actionable design principles for RNA-targeted ligand development.

## 📂 Repository Structure

```text
.
├── data/                       # Datasets used in the study
│   ├── external datasets/      # Independent validation sets (R-SIM, SM2miR)
│   ├── RNAdecoyDB/             # Our bias-matched hard-negative dataset
│   ├── ROBIN Positive/         # Positive samples from ROBIN
│   └── ROBIN Negative/         # Original negative samples 
│
├── TRN/                        # Source code for the Target RNA Network
│   ├── check point/            # Pre-trained model weights
│   ├── model/                  # Model architecture definitions
│   ├── Predict/                # Inference scripts
│   ├── Train/                  # Training scripts
│   └── yaml/                   # Configuration files 
│
└── zinc-TRN/                   # Screening results
    └── zincTRN.csv             # The filtered library of 6.96M RNA-binding molecules
```

## 🛠️ Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/TRN.git
cd TRN
```
**2. Create environment from YAML**

We provide a TRN.yaml file that contains all necessary dependencies.
```Bash
# Create the environment
conda env create -f TRN.yaml

# Activate the environment
conda activate TRN
```

## 🚀 Usage
**1. Training the TRN Model**

To train the model from scratch using RNAdecoyDB, navigate to the training directory. Ensure your configuration is set in the yaml folder.
```Bash
cd TRN/Train
python train_TRN.py
```
**2. Prediction**

To use the pre-trained model located in TRN/check point to predict the RNA-binding potential of new molecules:
```Bash
cd TRN/Predict
python predict.py 
```
## 📊 Datasets
**RNAdecoyDB**

Located in data/RNAdecoyDB. This dataset contains negative samples that are physicochemically matched to known RNA binders, preventing models from relying on simple property artifacts.

**External Validation**

Located in data/external datasets. Includes R-SIM and SM2miR datasets used to validate the generalization capability of TRN.
## 💊 Zinc-TRN Library
The zinc-TRN folder contains zincTRN.csv, a curated list of approximately 6.96 million molecules from the ZINC database identified by TRN as having high RNA-binding potential.

