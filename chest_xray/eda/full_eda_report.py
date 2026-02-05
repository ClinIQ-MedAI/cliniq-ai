#!/usr/bin/env python3
"""
Comprehensive EDA Report for NIH Chest X-ray14 Dataset
Generates detailed visualizations and insights
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/N/scratch/moabouag/cliniq/data/chest"
OUTPUT_DIR = "/N/u/moabouag/Quartz/Documents/cliniq/chest_xray/eda/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 70)
print("🫁 NIH CHEST X-RAY14 DATASET - COMPREHENSIVE EDA REPORT")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📂 Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "Data_Entry_2017.csv"))
bbox_df = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))

# Load train/test splits
with open(os.path.join(DATA_DIR, "train_val_list.txt"), 'r') as f:
    train_images = set(f.read().strip().split('\n'))
with open(os.path.join(DATA_DIR, "test_list.txt"), 'r') as f:
    test_images = set(f.read().strip().split('\n'))

df['split'] = df['Image Index'].apply(lambda x: 'train' if x in train_images else 'test')

print(f"✅ Loaded {len(df):,} images")

# ============================================================================
# 2. DATASET OVERVIEW
# ============================================================================
print("\n" + "=" * 70)
print("📊 DATASET OVERVIEW")
print("=" * 70)

overview_stats = {
    'Total Images': len(df),
    'Unique Patients': df['Patient ID'].nunique(),
    'Train Images': len(train_images),
    'Test Images': len(test_images),
    'Image Dimensions': f"{df['OriginalImage[Width'].median():.0f} x {df['Height]'].median():.0f}",
}

for key, val in overview_stats.items():
    print(f"  {key}: {val}")

# ============================================================================
# 3. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("📋 CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

# Extract all labels
all_labels = []
for labels in df['Finding Labels']:
    for label in labels.split('|'):
        all_labels.append(label.strip())

label_counts = Counter(all_labels)
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

# Create DataFrame for plotting
class_df = pd.DataFrame(sorted_labels, columns=['Class', 'Count'])
class_df['Percentage'] = class_df['Count'] / len(df) * 100

print("\nClass Distribution:")
print(class_df.to_string(index=False))

# FIGURE 1: Class Distribution Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = ['#e74c3c' if c == 'No Finding' else '#3498db' for c in class_df['Class']]
ax1 = axes[0]
bars = ax1.barh(class_df['Class'][::-1], class_df['Count'][::-1], color=colors[::-1])
ax1.set_xlabel('Number of Images', fontsize=12)
ax1.set_title('Class Distribution (Image Count)', fontsize=14, fontweight='bold')
ax1.axvline(x=class_df['Count'].mean(), color='red', linestyle='--', label=f'Mean: {class_df["Count"].mean():,.0f}')
ax1.legend()

# Add value labels
for bar, count in zip(bars, class_df['Count'][::-1]):
    ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
             f'{count:,}', va='center', fontsize=9)

# Pie chart for diseases only (excluding No Finding)
disease_df = class_df[class_df['Class'] != 'No Finding'].copy()
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(disease_df['Count'], labels=disease_df['Class'], 
                                    autopct='%1.1f%%', pctdistance=0.8)
ax2.set_title('Disease Distribution (Excluding "No Finding")', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_class_distribution.png")

# ============================================================================
# 4. CLASS IMBALANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("⚖️ CLASS IMBALANCE ANALYSIS")
print("=" * 70)

# Calculate imbalance ratio
max_count = class_df['Count'].max()
min_count = class_df[class_df['Class'] != 'No Finding']['Count'].min()
imbalance_ratio = max_count / min_count

print(f"  Maximum class count (No Finding): {max_count:,}")
print(f"  Minimum disease count (Hernia): {min_count:,}")
print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")

# FIGURE 2: Log-scale class distribution
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(class_df)), class_df['Count'], color=colors)
ax.set_yscale('log')
ax.set_xticks(range(len(class_df)))
ax.set_xticklabels(class_df['Class'], rotation=45, ha='right')
ax.set_ylabel('Count (log scale)', fontsize=12)
ax.set_title('Class Distribution - Log Scale (Shows Imbalance)', fontsize=14, fontweight='bold')
ax.axhline(y=class_df['Count'].median(), color='red', linestyle='--', label=f'Median: {class_df["Count"].median():,.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_class_imbalance_log.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_class_imbalance_log.png")

# ============================================================================
# 5. MULTI-LABEL ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("🏷️ MULTI-LABEL ANALYSIS")
print("=" * 70)

df['num_labels'] = df['Finding Labels'].apply(lambda x: len(x.split('|')))
label_dist = df['num_labels'].value_counts().sort_index()

print(f"\nLabels per Image Distribution:")
for num, count in label_dist.items():
    pct = count / len(df) * 100
    print(f"  {num} label(s): {count:,} images ({pct:.1f}%)")

print(f"\nStatistics:")
print(f"  Mean labels per image: {df['num_labels'].mean():.2f}")
print(f"  Max labels per image: {df['num_labels'].max()}")
print(f"  Single-label images: {(df['num_labels'] == 1).sum():,} ({(df['num_labels'] == 1).sum()/len(df)*100:.1f}%)")
print(f"  Multi-label images: {(df['num_labels'] > 1).sum():,} ({(df['num_labels'] > 1).sum()/len(df)*100:.1f}%)")

# FIGURE 3: Multi-label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
ax1.bar(label_dist.index, label_dist.values, color='#3498db', edgecolor='white')
ax1.set_xlabel('Number of Labels per Image', fontsize=12)
ax1.set_ylabel('Number of Images', fontsize=12)
ax1.set_title('Distribution of Labels per Image', fontsize=14, fontweight='bold')
for i, (num, count) in enumerate(label_dist.items()):
    ax1.text(num, count + 1000, f'{count:,}', ha='center', fontsize=9)

# Cumulative distribution
ax2 = axes[1]
cumsum = label_dist.cumsum() / len(df) * 100
ax2.plot(cumsum.index, cumsum.values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
ax2.fill_between(cumsum.index, cumsum.values, alpha=0.3, color='#e74c3c')
ax2.set_xlabel('Number of Labels', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Distribution of Labels', fontsize=14, fontweight='bold')
ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_multilabel_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_multilabel_distribution.png")

# ============================================================================
# 6. CO-OCCURRENCE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("🔗 DISEASE CO-OCCURRENCE ANALYSIS")
print("=" * 70)

# Create co-occurrence matrix
diseases = [l for l, _ in sorted_labels if l != 'No Finding']
cooccurrence = np.zeros((len(diseases), len(diseases)))

for labels in df['Finding Labels']:
    present = [l.strip() for l in labels.split('|') if l.strip() in diseases]
    for i, d1 in enumerate(diseases):
        for j, d2 in enumerate(diseases):
            if d1 in present and d2 in present:
                cooccurrence[i, j] += 1

# Normalize by diagonal (self-occurrence)
cooccurrence_pct = cooccurrence.copy()
for i in range(len(diseases)):
    if cooccurrence[i, i] > 0:
        cooccurrence_pct[i, :] = cooccurrence[i, :] / cooccurrence[i, i] * 100

# FIGURE 4: Co-occurrence heatmap
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(cooccurrence_pct, dtype=bool), k=1)
sns.heatmap(cooccurrence_pct, mask=mask, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=diseases, yticklabels=diseases, ax=ax,
            cbar_kws={'label': 'Co-occurrence %'})
ax.set_title('Disease Co-occurrence Matrix (%)\n(Row disease appears with column disease)', 
             fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_cooccurrence_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_cooccurrence_matrix.png")

# Top co-occurrences
print("\nTop 10 Disease Co-occurrences:")
pairs = []
for i in range(len(diseases)):
    for j in range(i+1, len(diseases)):
        pairs.append((diseases[i], diseases[j], cooccurrence[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)
for d1, d2, count in pairs[:10]:
    print(f"  {d1} + {d2}: {int(count):,} images")

# ============================================================================
# 7. PATIENT DEMOGRAPHICS
# ============================================================================
print("\n" + "=" * 70)
print("👤 PATIENT DEMOGRAPHICS")
print("=" * 70)

# Age analysis
df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce')
valid_ages = df[(df['Patient Age'] >= 0) & (df['Patient Age'] <= 100)]

print(f"\nAge Statistics:")
print(f"  Mean: {valid_ages['Patient Age'].mean():.1f} years")
print(f"  Median: {valid_ages['Patient Age'].median():.1f} years")
print(f"  Std Dev: {valid_ages['Patient Age'].std():.1f} years")
print(f"  Range: {valid_ages['Patient Age'].min():.0f} - {valid_ages['Patient Age'].max():.0f} years")

# Gender analysis
gender_counts = df['Patient Gender'].value_counts()
print(f"\nGender Distribution:")
for gender, count in gender_counts.items():
    print(f"  {gender}: {count:,} ({count/len(df)*100:.1f}%)")

# FIGURE 5: Demographics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age histogram
ax1 = axes[0, 0]
ax1.hist(valid_ages['Patient Age'], bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax1.axvline(x=valid_ages['Patient Age'].mean(), color='red', linestyle='--', 
            label=f'Mean: {valid_ages["Patient Age"].mean():.1f}')
ax1.axvline(x=valid_ages['Patient Age'].median(), color='orange', linestyle='--', 
            label=f'Median: {valid_ages["Patient Age"].median():.1f}')
ax1.set_xlabel('Age (years)', fontsize=12)
ax1.set_ylabel('Number of Images', fontsize=12)
ax1.set_title('Age Distribution', fontsize=14, fontweight='bold')
ax1.legend()

# Gender pie
ax2 = axes[0, 1]
colors_gender = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax2.pie(gender_counts.values, labels=gender_counts.index,
                                    autopct='%1.1f%%', colors=colors_gender, explode=(0.02, 0.02))
ax2.set_title('Gender Distribution', fontsize=14, fontweight='bold')

# Age by gender (box plot)
ax3 = axes[1, 0]
valid_ages.boxplot(column='Patient Age', by='Patient Gender', ax=ax3)
ax3.set_xlabel('Gender', fontsize=12)
ax3.set_ylabel('Age (years)', fontsize=12)
ax3.set_title('Age Distribution by Gender', fontsize=14, fontweight='bold')
plt.suptitle('')

# View position
ax4 = axes[1, 1]
view_counts = df['View Position'].value_counts()
ax4.bar(view_counts.index, view_counts.values, color=['#2ecc71', '#9b59b6'])
ax4.set_xlabel('View Position', fontsize=12)
ax4.set_ylabel('Number of Images', fontsize=12)
ax4.set_title('X-ray View Position Distribution', fontsize=14, fontweight='bold')
for i, (view, count) in enumerate(view_counts.items()):
    ax4.text(i, count + 1000, f'{count:,}\n({count/len(df)*100:.1f}%)', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_demographics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 05_demographics.png")

# ============================================================================
# 8. DISEASE BY DEMOGRAPHICS
# ============================================================================
print("\n" + "=" * 70)
print("📊 DISEASE PREVALENCE BY DEMOGRAPHICS")
print("=" * 70)

# Create binary columns for each disease
for disease in diseases:
    df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

# Disease by gender
disease_by_gender = df.groupby('Patient Gender')[diseases].mean() * 100

# FIGURE 6: Disease by gender
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(diseases))
width = 0.35
bars1 = ax.bar(x - width/2, disease_by_gender.loc['M'], width, label='Male', color='#3498db')
bars2 = ax.bar(x + width/2, disease_by_gender.loc['F'], width, label='Female', color='#e74c3c')
ax.set_ylabel('Prevalence (%)', fontsize=12)
ax.set_xlabel('Disease', fontsize=12)
ax.set_title('Disease Prevalence by Gender', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(diseases, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_disease_by_gender.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 06_disease_by_gender.png")

# Disease by age group
df['Age Group'] = pd.cut(df['Patient Age'], bins=[0, 20, 40, 60, 80, 100], 
                         labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
disease_by_age = df.groupby('Age Group')[diseases].mean() * 100

# FIGURE 7: Disease by age
fig, ax = plt.subplots(figsize=(14, 8))
disease_by_age.T.plot(kind='bar', ax=ax, width=0.8)
ax.set_ylabel('Prevalence (%)', fontsize=12)
ax.set_xlabel('Disease', fontsize=12)
ax.set_title('Disease Prevalence by Age Group', fontsize=14, fontweight='bold')
ax.legend(title='Age Group', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_disease_by_age.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_disease_by_age.png")

# ============================================================================
# 9. TRAIN/TEST SPLIT ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("📁 TRAIN/TEST SPLIT ANALYSIS")
print("=" * 70)

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

print(f"\nSplit Statistics:")
print(f"  Train: {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Test: {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%)")

# Class distribution in train vs test
train_dist = train_df['Finding Labels'].apply(lambda x: x.split('|')[0]).value_counts(normalize=True) * 100
test_dist = test_df['Finding Labels'].apply(lambda x: x.split('|')[0]).value_counts(normalize=True) * 100

# FIGURE 8: Train vs Test distribution
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(class_df))
width = 0.35

train_pcts = [train_df[train_df['Finding Labels'].str.contains(c)].shape[0] / len(train_df) * 100 
              if c != 'No Finding' else 
              train_df[train_df['Finding Labels'] == 'No Finding'].shape[0] / len(train_df) * 100
              for c in class_df['Class']]
test_pcts = [test_df[test_df['Finding Labels'].str.contains(c)].shape[0] / len(test_df) * 100 
             if c != 'No Finding' else 
             test_df[test_df['Finding Labels'] == 'No Finding'].shape[0] / len(test_df) * 100
             for c in class_df['Class']]

bars1 = ax.bar(x - width/2, train_pcts, width, label='Train', color='#3498db')
bars2 = ax.bar(x + width/2, test_pcts, width, label='Test', color='#e74c3c')
ax.set_ylabel('Prevalence (%)', fontsize=12)
ax.set_xlabel('Class', fontsize=12)
ax.set_title('Class Distribution: Train vs Test Split', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_df['Class'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_train_test_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 08_train_test_distribution.png")

# ============================================================================
# 10. BOUNDING BOX ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("📦 BOUNDING BOX ANALYSIS")
print("=" * 70)

print(f"\nBounding Box Statistics:")
print(f"  Total Annotations: {len(bbox_df):,}")
print(f"  Images with BBox: {bbox_df['Image Index'].nunique():,}")
print(f"  Coverage: {len(bbox_df)/len(df)*100:.2f}% of images")

bbox_class_counts = bbox_df['Finding Label'].value_counts()
print(f"\nBBox per Class:")
for cls, count in bbox_class_counts.items():
    print(f"  {cls}: {count:,}")

# FIGURE 9: BBox analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BBox count per class
ax1 = axes[0]
ax1.barh(bbox_class_counts.index[::-1], bbox_class_counts.values[::-1], color='#2ecc71')
ax1.set_xlabel('Number of Bounding Boxes', fontsize=12)
ax1.set_title('Bounding Box Annotations per Class', fontsize=14, fontweight='bold')

# BBox size distribution
if 'Bbox [x' in bbox_df.columns:
    bbox_df['width'] = bbox_df.iloc[:, 4]
    bbox_df['height'] = bbox_df.iloc[:, 5]
    bbox_df['area'] = bbox_df['width'] * bbox_df['height']
    
    ax2 = axes[1]
    ax2.hist(bbox_df['area'], bins=50, color='#9b59b6', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Bounding Box Area (pixels²)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Bounding Box Size Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_bbox_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 09_bbox_analysis.png")

# ============================================================================
# 11. IMAGE PROPERTIES
# ============================================================================
print("\n" + "=" * 70)
print("🖼️ IMAGE PROPERTIES")
print("=" * 70)

print(f"\nImage Dimensions:")
print(f"  Width - Mean: {df['OriginalImage[Width'].mean():.0f}, Range: {df['OriginalImage[Width'].min():.0f} - {df['OriginalImage[Width'].max():.0f}")
print(f"  Height - Mean: {df['Height]'].mean():.0f}, Range: {df['Height]'].min():.0f} - {df['Height]'].max():.0f}")

# FIGURE 10: Image dimensions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist2d(df['OriginalImage[Width'], df['Height]'], bins=50, cmap='YlOrRd')
ax1.set_xlabel('Width (pixels)', fontsize=12)
ax1.set_ylabel('Height (pixels)', fontsize=12)
ax1.set_title('Image Dimension Distribution', fontsize=14, fontweight='bold')
plt.colorbar(ax1.collections[0], ax=ax1, label='Count')

# Pixel spacing
ax2 = axes[1]
df['pixel_spacing'] = df['OriginalImagePixelSpacing[x'].astype(float)
ax2.hist(df['pixel_spacing'], bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax2.set_xlabel('Pixel Spacing (mm)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Pixel Spacing Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_image_properties.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 10_image_properties.png")

# ============================================================================
# 12. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("💡 KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

insights = """
📌 KEY INSIGHTS:

1. SEVERE CLASS IMBALANCE
   - "No Finding" dominates with 53.8% of images
   - Hernia is extremely rare (0.2%, only 227 cases)
   - Imbalance ratio: ~266:1 between largest and smallest class
   → Recommendation: Use weighted loss, oversampling, or focal loss

2. MULTI-LABEL NATURE
   - 18.5% of images have multiple diseases
   - Up to 9 diseases can co-occur in single image
   - Infiltration frequently co-occurs with other diseases
   → Recommendation: Use multi-label BCE loss, not softmax

3. DISEASE CO-OCCURRENCE PATTERNS
   - Infiltration + Effusion: Most common combination
   - Atelectasis often appears with Effusion
   - Emphysema + Pneumothorax show high co-occurrence
   → Insight: Consider disease relationships in model design

4. DEMOGRAPHIC PATTERNS
   - Slight male predominance (56.5%)
   - Mean age ~47 years, mostly middle-aged adults
   - Some diseases show clear gender differences
   → Consideration: Demographics could be useful features

5. LIMITED LOCALIZATION DATA
   - Only 984 bounding boxes (<1% coverage)
   - Only 8 disease classes have annotations
   → Conclusion: Not suitable for object detection (YOLO)

6. TRAIN/TEST SPLIT
   - 77% train, 23% test (pre-defined)
   - Similar class distributions in both splits
   → Use provided split for fair comparison with literature

🎯 RECOMMENDED APPROACH:
   - Model: DenseNet-121 or ConvNeXt
   - Loss: Weighted BCE or Focal Loss
   - Output: Sigmoid (not softmax) for multi-label
   - Metric: AUC-ROC (industry standard for this dataset)
   - Handle imbalance: Class weights + data augmentation
"""

print(insights)

# Save insights to file
with open(os.path.join(OUTPUT_DIR, 'insights.txt'), 'w') as f:
    f.write(insights)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ EDA REPORT COMPLETE")
print("=" * 70)
print(f"\n📁 All figures saved to: {OUTPUT_DIR}")
print("\nGenerated Visualizations:")
for i, fname in enumerate(sorted(os.listdir(OUTPUT_DIR)), 1):
    if fname.endswith('.png'):
        print(f"  {i}. {fname}")

print("\n🏥 Dataset ready for multi-label classification training!")
