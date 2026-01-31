# charset.py
# Build and manage charset for recognition.

from __future__ import annotations
import string
from typing import Dict, List, Tuple

def default_charset() -> str:
    # You can replace this with a dataset-derived charset if needed.
    return string.ascii_letters + string.digits + string.punctuation + " "

def build_vocab(charset: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    # CTC requires blank token at index 0
    char2id = {"<BLANK>": 0}
    for i, c in enumerate(charset):
        if c not in char2id:
            char2id[c] = i + 1
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char
