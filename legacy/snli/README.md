# PyTorch-based NLI Training with SNLI

## 📝 Overview

This repository contains Python scripts to train a Natural Language Inference (NLI) model, specifically the `SNLIClassifier`, using the Stanford Natural Language Inference (SNLI) corpus. The trained model predicts textual entailment, identifying if a statement is entailed, contradicted, or neither by another statement. Tokenization uses **spacy** (no torchtext).

## ⚙️ Dependencies

Install the necessary Python libraries with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The `requirements.txt` file includes:

```
torch
spacy
datasets
```

## 💻 Usage

Start the training process with:

```bash
python train.py --lower --epochs [NUMBER_OF_EPOCHS] --batch_size [BATCH_SIZE] --save-path [PATH_TO_SAVE_MODEL] --gpu [GPU_NUMBER]
```

Optional: `--word_vectors [PATH]` and `--vector_cache [PATH]` for pretrained word vectors (e.g. GloVe).

## 🏋️‍♀️ Training

The script trains the model on mini-batches of data across a specified number of epochs. It saves the best-performing model on the validation set as a `.pt` file in the specified directory. SNLI is loaded via the HuggingFace `datasets` library.

## 📚 Scripts

- `model.py`: Defines the `SNLIClassifier` model and auxiliary classes.
- `data.py`: SNLI loading (HuggingFace datasets), spacy tokenization, and vocabulary.
- `util.py`: Utility functions for directory creation and command-line argument parsing.

## 📣 Note

Ensure the `model.py`, `data.py`, and `util.py` scripts are available in your working directory.
