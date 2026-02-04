import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import spacy
from collections import Counter
import tarfile
import urllib.request
import os
from pathlib import Path

# Custom Vocabulary class to replace torchtext.vocab
class Vocab:
    def __init__(self, counter, specials, min_freq=1):
        self.itos = list(specials.keys())  # index to string
        self.stoi = dict(specials)  # string to index
        self.default_index = specials.get("<unk>", 0)
        
        # Add words from counter that meet min_freq threshold
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def __call__(self, tokens):
        """Convert a list of tokens to indices"""
        return [self[token] for token in tokens]
    
    def set_default_index(self, index):
        self.default_index = index
    
    def lookup_tokens(self, indices):
        """Convert indices back to tokens"""
        return [self.itos[idx] if idx < len(self.itos) else "<unk>" for idx in indices]


# Multi30k dataset - download and parse manually
class Multi30kDataset(Dataset):
    URLS = {
        "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    }
    
    def __init__(self, split, language_pair, root=".data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.src_lang, self.tgt_lang = language_pair
        
        # Download and extract if needed
        data_file = self.root / f"multi30k_{split}.{self.src_lang}-{self.tgt_lang}"
        if not data_file.exists():
            self._download_and_extract(split)
        
        # Load the data
        self.data = self._load_data(split)
    
    def _download_and_extract(self, split):
        url = self.URLS[split]
        tar_path = self.root / f"multi30k_{split}.tar.gz"
        
        # Download
        print(f"Downloading Multi30k {split} data...")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        print(f"Extracting Multi30k {split} data...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.root)
        
        # Clean up tar file
        tar_path.unlink()
    
    def _load_data(self, split):
        # File naming convention in Multi30k
        if split == "train":
            prefix = "train"
        else:
            prefix = "val"
        
        src_file = self.root / f"{prefix}.{self.src_lang}"
        tgt_file = self.root / f"{prefix}.{self.tgt_lang}"
        
        with open(src_file, "r", encoding="utf-8") as f:
            src_lines = f.read().strip().split("\n")
        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_lines = f.read().strip().split("\n")
        
        return list(zip(src_lines, tgt_lines))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Spacy tokenizer wrapper
class SpacyTokenizer:
    # Map language codes to spacy model names
    LANG_MAP = {
        "de": "de_core_news_sm",
        "en": "en_core_web_sm",
    }
    
    def __init__(self, lang):
        model_name = self.LANG_MAP.get(lang, f"{lang}_core_news_sm")
        self.nlp = spacy.load(model_name, disable=["parser", "ner", "tagger"])
    
    def __call__(self, text):
        return [token.text.lower() for token in self.nlp(text)]


def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    """Build vocabulary from an iterator of token lists"""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    
    special_dict = {s: i for i, s in enumerate(specials)} if specials else {}
    return Vocab(counter, special_dict, min_freq)


# Turns an iterable into a generator
def _yield_tokens(iterable_data, tokenizer, src):
    # Iterable data stores the samples as (src, tgt) so this will help us select just one language or the other
    index = 0 if src else 1

    for data in iterable_data:
        yield tokenizer(data[index])


# Get data, tokenizer, text transform, vocab objs, etc. Everything we
# need to start training the model
def get_data(opts):

    src_lang = opts.src
    tgt_lang = opts.tgt

    # Define a token "unknown", "padding", "beginning of sentence", and "end of sentence"
    special_symbols = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3
    }

    # Get training examples from Multi30k dataset
    train_dataset = Multi30kDataset(split="train", language_pair=(src_lang, tgt_lang))
    valid_dataset = Multi30kDataset(split="valid", language_pair=(src_lang, tgt_lang))

    # Grab a tokenizer for these languages
    src_tokenizer = SpacyTokenizer(src_lang)
    tgt_tokenizer = SpacyTokenizer(tgt_lang)

    # Build a vocabulary object for these languages
    src_vocab = build_vocab_from_iterator(
        _yield_tokens(train_dataset, src_tokenizer, True),
        min_freq=1,
        specials=list(special_symbols.keys()),
    )

    tgt_vocab = build_vocab_from_iterator(
        _yield_tokens(train_dataset, tgt_tokenizer, False),
        min_freq=1,
        specials=list(special_symbols.keys()),
    )

    src_vocab.set_default_index(special_symbols["<unk>"])
    tgt_vocab.set_default_index(special_symbols["<unk>"])

    # Helper function to sequentially apply transformations
    def _seq_transform(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # Function to add BOS/EOS and create tensor for input sequence indices
    def _tensor_transform(token_ids):
        return torch.cat(
            (torch.tensor([special_symbols["<bos>"]]),
             torch.tensor(token_ids),
             torch.tensor([special_symbols["<eos>"]]))
        )

    src_lang_transform = _seq_transform(src_tokenizer, src_vocab, _tensor_transform)
    tgt_lang_transform = _seq_transform(tgt_tokenizer, tgt_vocab, _tensor_transform)

    # Now we want to convert the data to a dataloader. We
    # will need to collate batches
    def _collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])
        return src_batch, tgt_batch

    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch, collate_fn=_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opts.batch, collate_fn=_collate_fn)

    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols


def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):

    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# A small test to make sure our data loads in correctly
if __name__ == "__main__":

    class Opts:
        def __init__(self):
            self.src = "de"
            self.tgt = "en"
            self.batch = 128

    opts = Opts()
    
    train_dl, valid_dl, src_vocab, tgt_vocab, src_lang_transform, tgt_lang_transform, special_symbols = get_data(opts)

    print(f"{opts.src} vocab size: {len(src_vocab)}")
    print(f"{opts.tgt} vocab size: {len(tgt_vocab)}")
