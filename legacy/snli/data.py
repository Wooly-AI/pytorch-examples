"""SNLI data loading with spacy tokenization (no torchtext)."""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy


class Vocab:
    """Vocabulary: map tokens to indices (replaces torchtext.vocab)."""
    def __init__(self, counter, specials, min_freq=1):
        self.itos = list(specials.keys())
        self.stoi = dict(specials)
        self.default_index = specials.get("<unk>", 0)
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def set_default_index(self, index):
        self.default_index = index


class SpacyTokenizer:
    """Tokenize with spacy (lowercase optional)."""
    def __init__(self, model_name="en_core_web_sm", lower=True):
        self.nlp = spacy.load(model_name, disable=["parser", "ner", "tagger"])
        self.lower = lower

    def __call__(self, text):
        if self.lower:
            return [t.text.lower() for t in self.nlp(text)]
        return [t.text for t in self.nlp(text)]


class SNLIDataset(Dataset):
    """SNLI from HuggingFace datasets."""
    LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(self, split, tokenizer, max_examples=None):
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/snli", split=split)
        self.examples = []
        for i, row in enumerate(ds):
            if max_examples is not None and i >= max_examples:
                break
            label = row["label"]
            if label == -1:
                continue
            premise = tokenizer(row["premise"])
            hypothesis = tokenizer(row["hypothesis"])
            self.examples.append((premise, hypothesis, label))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_vocab_from_datasets(train_ds, specials=None):
    counter = Counter()
    for premise_tok, hypothesis_tok, _ in train_ds.examples:
        counter.update(premise_tok)
        counter.update(hypothesis_tok)
    special_dict = {s: i for i, s in enumerate(specials)} if specials else {}
    return Vocab(counter, special_dict, min_freq=1)


class Batch:
    """Batch with .premise, .hypothesis, .label, .batch_size (model expects these)."""
    __slots__ = ("premise", "hypothesis", "label", "batch_size")

    def __init__(self, premise, hypothesis, label):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label
        self.batch_size = label.size(0)


def get_data(args, max_train_examples=None, max_dev_examples=None):
    """Load SNLI, build vocab, return train/dev/test loaders and config updates."""
    tokenizer = SpacyTokenizer(lower=args.lower)
    train_ds = SNLIDataset("train", tokenizer, max_examples=max_train_examples)
    dev_ds = SNLIDataset("validation", tokenizer, max_examples=max_dev_examples)
    test_ds = SNLIDataset("test", tokenizer, max_examples=max_dev_examples)

    specials = ["<unk>", "<pad>"]
    input_vocab = build_vocab_from_datasets(train_ds, specials=specials)
    input_vocab.set_default_index(specials.index("<unk>"))
    pad_idx = specials.index("<pad>")
    answer_vocab_size = 3  # entailment, neutral, contradiction

    def collate(batch_list):
        premises = [torch.tensor(input_vocab(p), dtype=torch.long) for p, h, l in batch_list]
        hypotheses = [torch.tensor(input_vocab(h), dtype=torch.long) for p, h, l in batch_list]
        labels = torch.tensor([l for _, _, l in batch_list], dtype=torch.long)
        premises_padded = pad_sequence(premises, padding_value=pad_idx)
        hypotheses_padded = pad_sequence(hypotheses, padding_value=pad_idx)
        return Batch(premises_padded, hypotheses_padded, labels)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    return train_loader, dev_loader, test_loader, input_vocab, answer_vocab_size, len(dev_ds)
