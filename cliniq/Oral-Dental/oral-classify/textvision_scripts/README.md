# TextVision (Scripts)

This folder contains a script-based version of the original notebook pipeline.

## 1) Build word crops dataset
1. Ensure you have the raw train folder containing `*.jpg` and `*.json`.
2. Run:
   - `python build_word_dataset.py --overwrite`
   - `python split_data.py`

## 2) Train CRNN + CTC
`python train.py --epochs 30 --batch_size 64`

Checkpoints are saved to `/content/checkpoints` by default.

## 3) Quick evaluation
`python eval_samples.py --ckpt /content/checkpoints/crnn_epoch030.pth`

## 4) Single image inference
`python infer.py --ckpt /content/checkpoints/crnn_epoch030.pth --image /content/word_dataset/images/<file>.jpg`

## Notes
- This keeps the same logic as the notebook but in clean modules.
- All code comments are in English.
