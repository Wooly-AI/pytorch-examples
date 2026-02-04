import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from model import SNLIClassifier
from util import get_args, makedirs
from data import get_data


args = get_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

max_train = (args.batch_size * 2) if args.dry_run else None
max_dev = (args.batch_size * 2) if args.dry_run else None
train_loader, dev_loader, test_loader, input_vocab, answer_vocab_size, dev_len = get_data(
    args, max_train_examples=max_train, max_dev_examples=max_dev
)

config = args
config.n_embed = len(input_vocab)
config.d_out = answer_vocab_size
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
else:
    model = SNLIClassifier(config)
    if getattr(args, 'word_vectors', None) and args.word_vectors:
        if os.path.isfile(args.vector_cache):
            vectors = torch.load(args.vector_cache)
            model.embed.weight.data.copy_(vectors)
    model.to(device)

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
makedirs(args.save_path)


def log_progress(epoch, iterations, batch_idx, n_batches, loss, train_acc, dev_loss=None, dev_acc=None):
    """Print progress as Title: Value lines for easy pattern matching."""
    pct = 100. * (1 + batch_idx) / n_batches if n_batches else 0
    print(f"Iteration: {iterations}")
    print(f"Epoch: {epoch}")
    print(f"Progress: {1 + batch_idx}/{n_batches} ({pct:.0f}%)")
    print(f"Loss: {loss:.6f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    if dev_loss is not None:
        print(f"Dev Loss: {dev_loss:.6f}")
    if dev_acc is not None:
        print(f"Dev Accuracy: {dev_acc:.4f}")

for epoch in range(args.epochs):
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_loader):
        batch.premise = batch.premise.to(device)
        batch.hypothesis = batch.hypothesis.to(device)
        batch.label = batch.label.to(device)

        model.train()
        opt.zero_grad()
        iterations += 1

        answer = model(batch)
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(answer, batch.label)
        loss.backward()
        opt.step()

        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(
                train_acc, loss.item(), iterations
            )
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        if iterations % args.dev_every == 0:
            model.eval()
            n_dev_correct, dev_loss_sum = 0, 0.0
            with torch.no_grad():
                for dev_batch in dev_loader:
                    dev_batch.premise = dev_batch.premise.to(device)
                    dev_batch.hypothesis = dev_batch.hypothesis.to(device)
                    dev_batch.label = dev_batch.label.to(device)
                    answer = model(dev_batch)
                    n_dev_correct += (
                        torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label
                    ).sum().item()
                    dev_loss_sum += criterion(answer, dev_batch.label).item()
            dev_acc = 100. * n_dev_correct / dev_len
            dev_loss_val = dev_loss_sum / max(len(dev_loader), 1)

            log_progress(epoch, iterations, batch_idx, len(train_loader), loss.item(),
                         train_acc, dev_loss=dev_loss_val, dev_acc=dev_acc)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
                    dev_acc, dev_loss_val, iterations
                )
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:
            log_progress(epoch, iterations, batch_idx, len(train_loader), loss.item(), train_acc)
        if args.dry_run:
            if iterations % args.log_every != 0 and iterations % args.dev_every != 0:
                log_progress(epoch, iterations, batch_idx, len(train_loader), loss.item(), train_acc)
            break
    if args.dry_run:
        break
