# CATH protein structure classification

This repository contains code for the ML protein structure classification problem solution as a part of ML hands-on challenge.

## Setup

First of all, you need to install the required packages. You can do this by running the following command:

```bash
# Install conda environment
conda create -n ml-challenge python=3.10
conda activate ml-challenge

# Install required packages
pip install -r requirements.txt
```

## Running
To run model inference on protein sequences please use `scripts/main.py`:

```bash
> python scripts/main.py --help
usage: main.py [-h] -i INPUT_CSV [-m MODEL_CHECKPOINT] [-o OUTPUT_CSV]

Protein Classification

options:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input_csv INPUT_CSV
                        Input CSV file containing new sequences
  -m MODEL_CHECKPOINT, --model_checkpoint MODEL_CHECKPOINT
                        Path to the model checkpoint
  -o OUTPUT_CSV, --output_csv OUTPUT_CSV
                        Output CSV file to save predictions
```

Here `-m` and `-o` parameters are **optional** (the former points to `data/model_best.pt` and the latter points to saving output in the same folder).

***NOTE***: **The input CSV file should contain a `sequence` column with protein sequences!**

Some examples of the run:

```bash
# it will load `data/model_best.pt` model and save results to `data/data_processed_with_predictions.csv`
python scripts/main.py -i data/data_processed.csv

# it will load `data/model_epoch_10.pt` model and save results to `data/inferred_data.csv`
python scripts/main.py -i data/data_processed.csv -o data/inferred_data.csv

# it will load `data/model_epoch_10.pt` model and save results to `data/data_processed_with_predictions.csv`
python scripts/main.py -i data/data_processed.csv -m data/model_epoch_10.pt
```

## Data

As input to the project we obtained `pdb_share.zip` file of protein sequences with corresponding CATH labels along with `cath_w_seqs_share.csv` file. We processed these two files and obtained `data_processed.csv`. To obtain it please do:

- Create a `data` directory in the root of the repository
- Unzip the `pdb_share.zip` file into the `data` directory:
- Move `cath_w_seqs_share.csv` into the `data` directory
- Run `data_prep_exploration.ipynb` notebook

## Repository overview

The repository is organized as follows:

- `data` - directory with the input data and the processed data
  - `data_processed.csv` - file with the processed data
  - `model_epoch_[10,20,30].pt` - model checkpoints
  - `model_best.pt` - best protiei classification model so far, trained on ESMModel embeddings
- `notebooks` - directory with the Jupyter notebooks
  - `data_prep_exploration.ipynb` - notebook with the data exploration and preparation, it will produce the `data_processed.csv` that we will use in downstream notebooks
  - `n_gram_count_model.ipynb` - notebook with the n-gram count model implementation, performs worse than embedding model
  - `embedding_model.ipynb` - notebook with the embedding model implementation, achieves best performance on the datase
  - `ML_challenge.ipynb` - playground notebook obtained from the team
- `scripts` - directory with the scripts
  - `main.py` - script for model inference on protein sequences