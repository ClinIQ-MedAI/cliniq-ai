# infer.py
# Run single-image inference using a saved checkpoint.

from __future__ import annotations
import argparse
import torch
from PIL import Image
from dataset import default_transform
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
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    char2id = ckpt["char2id"]
    id2char = ckpt["id2char"]
    num_classes = len(char2id)

    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = default_transform()
    img = Image.open(args.image).convert("L")
    x = transform(img).unsqueeze(0).to(device)

    preds = model(x)                 # [B, W, C]
    preds = preds.log_softmax(dim=-1)
    preds = preds.permute(1, 0, 2)   # [T, B, C]
    text = ctc_decode(preds[:,0,:], id2char, blank_id=0)

    print(text)

if __name__ == "__main__":
    main()
