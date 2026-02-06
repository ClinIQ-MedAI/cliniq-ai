# eval_samples.py
# Print a few predictions vs ground truth from the validation loader.

from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from config import Paths, Training
from dataset import OCRWordDataset, default_transform
from model import CRNN

def ctc_decode(preds_T_C, id2char, blank_id=0) -> str:
    idxs = torch.argmax(preds_T_C, dim=-1).tolist()
    out=[]
    prev=None
    for i in idxs:
        if i != blank_id and i != prev:
            out.append(id2char.get(i, ""))
        prev = i
    return "".join(out)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=Paths().data_root)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_batches", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    paths = Paths(data_root=args.root)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    char2id = ckpt["char2id"]
    id2char = ckpt["id2char"]
    num_classes = len(char2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = default_transform(Training().img_h, Training().img_w)
    val_ds = OCRWordDataset(paths.val_list, paths.word_images_dir, transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=Training().num_workers, pin_memory=Training().pin_memory)

    for i, (imgs, texts) in enumerate(val_loader):
        if i >= args.n_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs).log_softmax(-1).permute(1,0,2)  # [T,B,C]
        for b in range(min(imgs.size(0), 16)):
            pred_text = ctc_decode(preds[:,b,:], id2char, blank_id=0)
            print(f"GT: {texts[b]} | Pred: {pred_text}")

if __name__ == "__main__":
    main()
