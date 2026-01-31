import pandas as pd


csv_path = "/home/moabouag/far/oral-detect/oral_runs/YOLOv11x_ORAL_SOTA_20251125_0709/results.csv"   # أو trainXX

df = pd.read_csv(csv_path)

best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]

print("Best overall result in the run:")
print(best_row[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']].round(4))


print("\n best 5 results in the run sorted by mAP50:")
print(df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',]].sort_values('metrics/mAP50(B)', ascending=False).head().round(4))