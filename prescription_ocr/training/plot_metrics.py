import json
import matplotlib.pyplot as plt
import numpy as np

# Load logs
try:
    with open('logs/training_logs.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: logs/training_logs.json not found.")
    exit(1)

history = data['log_history']

# Extract data
train_steps = []
train_loss = []
eval_steps = []
eval_loss = []
eval_cer = []

for entry in history:
    if 'loss' in entry and 'eval_loss' not in entry:
        train_steps.append(entry['step'])
        train_loss.append(entry['loss'])
    elif 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_loss.append(entry['eval_loss'])
        eval_cer.append(entry['eval_cer'])

# Plot 1: Training vs Validation Loss
plt.figure(figsize=(12, 5))
plt.plot(train_steps, train_loss, label='Training Loss', alpha=0.6)
plt.plot(eval_steps, eval_loss, label='Validation Loss', linewidth=2, color='orange')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('logs/loss_plot_v2.png') # saving as v2 to avoid name collision if any
print("Saved logs/loss_plot_v2.png")

# Plot 2: Character Error Rate (CER)
plt.figure(figsize=(12, 5))
plt.plot(eval_steps, eval_cer, label='Validation CER', color='green', marker='o', markersize=4)
plt.xlabel('Steps')
plt.ylabel('CER')
plt.title('Character Error Rate (Lower is Better)')
plt.grid(True, alpha=0.3)

# Annotate best point
min_cer = min(eval_cer)
min_idx = eval_cer.index(min_cer)
best_step = eval_steps[min_idx]
plt.annotate(f'Best: {min_cer:.4f}', 
             xy=(best_step, min_cer), 
             xytext=(best_step, min_cer + 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('logs/cer_plot_v2.png')
print(f"Saved cer_plot_v2.png (Best CER: {min_cer:.4f} at step {best_step})")
