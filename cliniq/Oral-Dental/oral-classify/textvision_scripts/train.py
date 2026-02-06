# train.py
# Train CRNN+CTC on word crops.

from __future__ import annotations
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Paths, Training
from utils import seed_everything
from dataset import OCRWordDataset, default_transform
from charset import default_charset, build_vocab
from model import CRNN

def ctc_greedy_decode(log_probs_TBC, id2char, blank_id=0) -> str:
    # log_probs_TBC: [T, C] or [T, B, C]
    if log_probs_TBC.dim() == 3:
        log_probs_TBC = log_probs_TBC[:, 0, :]
    idxs = torch.argmax(log_probs_TBC, dim=-1).tolist()
    out=[]
    prev=None
    for i in idxs:
        if i != blank_id and i != prev:
            out.append(id2char.get(i, ""))
        prev = i
    return "".join(out)

def train_one_epoch(model, loader, optimizer, ctc_loss, device, char2id):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for imgs, texts in loader:
        imgs = imgs.to(device, non_blocking=True)

        preds = model(imgs)                 # [B, W, C]
        preds = preds.log_softmax(dim=-1)   # required for CTCLoss
        preds = preds.permute(1, 0, 2)      # [T, B, C]
        T = preds.size(0)

        all_targets=[]
        target_lengths=[]
        keep=[]

        for i, t in enumerate(texts):
            seq = [char2id[c] for c in t if c in char2id]
            if len(seq) == 0 or len(seq) > T:
                continue
            keep.append(i)
            all_targets.extend(seq)
            target_lengths.append(len(seq))

        if len(keep) == 0:
            continue

        preds = preds[:, keep, :]
        labels = torch.tensor(all_targets, dtype=torch.long, device=device)
        label_lengths = torch.tensor(target_lengths, dtype=torch.long, device="cpu")
        input_lengths = torch.full((len(keep),), preds.size(0), dtype=torch.long, device="cpu")

        loss = ctc_loss(preds, labels, input_lengths, label_lengths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)

@torch.no_grad()
def validate(model, loader, ctc_loss, device, char2id):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for imgs, texts in loader:
        imgs = imgs.to(device, non_blocking=True)

        preds = model(imgs)
        preds = preds.log_softmax(dim=-1)
        preds = preds.permute(1, 0, 2)
        T = preds.size(0)

        all_targets=[]
        target_lengths=[]
        keep=[]

        for i, t in enumerate(texts):
            seq = [char2id[c] for c in t if c in char2id]
            if len(seq) == 0 or len(seq) > T:
                continue
            keep.append(i)
            all_targets.extend(seq)
            target_lengths.append(len(seq))

        if len(keep) == 0:
            continue

        preds = preds[:, keep, :]
        labels = torch.tensor(all_targets, dtype=torch.long, device=device)
        label_lengths = torch.tensor(target_lengths, dtype=torch.long, device="cpu")
        input_lengths = torch.full((len(keep),), preds.size(0), dtype=torch.long, device="cpu")

        loss = ctc_loss(preds, labels, input_lengths, label_lengths)
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=Paths().data_root)
    p.add_argument("--epochs", type=int, default=Training().epochs)
    p.add_argument("--batch_size", type=int, default=Training().batch_size)
    p.add_argument("--lr", type=float, default=Training().lr)
    p.add_argument("--save_dir", type=str, default=Training().save_dir)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)

    paths = Paths(data_root=args.root)
    os.makedirs(args.save_dir, exist_ok=True)

    charset = default_charset()
    char2id, id2char = build_vocab(charset)
    num_classes = len(char2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = default_transform(Training().img_h, Training().img_w)
    train_ds = OCRWordDataset(paths.train_list, paths.word_images_dir, transform)
    val_ds   = OCRWordDataset(paths.val_list, paths.word_images_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=Training().num_workers, pin_memory=Training().pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=Training().num_workers, pin_memory=Training().pin_memory)

    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, ctc_loss, device, char2id)
        va = validate(model, val_loader, ctc_loss, device, char2id)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")

        ckpt = os.path.join(args.save_dir, f"crnn_epoch{epoch:03d}.pth")
        torch.save({
            "model": model.state_dict(),
            "char2id": char2id,
            "id2char": id2char,
            "charset": charset,
        }, ckpt)

if __name__ == "__main__":
    main()
