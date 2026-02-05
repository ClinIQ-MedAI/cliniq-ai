#!/usr/bin/env python3
"""
Generate PDF Report by combining all figures with text
"""

import os
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Paths
EDA_DIR = "/N/u/moabouag/Quartz/Documents/cliniq/chest_xray/eda"
FIGURES_DIR = os.path.join(EDA_DIR, "figures")
PDF_FILE = os.path.join(EDA_DIR, "EDA_REPORT.pdf")

# Create PDF
doc = SimpleDocTemplate(PDF_FILE, pagesize=A4, 
                        rightMargin=1.5*cm, leftMargin=1.5*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, spaceAfter=20, textColor=colors.HexColor('#2c3e50'))
heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=16, spaceAfter=10, spaceBefore=20, textColor=colors.HexColor('#2980b9'))
subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading2'], fontSize=13, spaceAfter=8, textColor=colors.HexColor('#34495e'))
body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=8, leading=14)
bold_style = ParagraphStyle('Bold', parent=body_style, fontName='Helvetica-Bold')

# Content
story = []

# Title
story.append(Paragraph("🫁 NIH Chest X-ray14 Dataset", title_style))
story.append(Paragraph("Exploratory Data Analysis Report", styles['Heading2']))
story.append(Spacer(1, 0.5*inch))

# Overview
story.append(Paragraph("📊 Dataset Overview", heading_style))
overview_data = [
    ['Metric', 'Value'],
    ['Total Images', '112,120'],
    ['Unique Patients', '30,805'],
    ['Train Set', '86,524 (77.2%)'],
    ['Test Set', '25,596 (22.8%)'],
    ['Image Dimensions', '~2518 × 2544 px'],
    ['Multi-label Images', '20,796 (18.5%)'],
]
t = Table(overview_data, colWidths=[3*inch, 3*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
]))
story.append(t)
story.append(Spacer(1, 0.3*inch))

# Class Distribution
story.append(Paragraph("📋 Class Distribution", heading_style))
img_path = os.path.join(FIGURES_DIR, '01_class_distribution.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=3*inch))
story.append(Spacer(1, 0.2*inch))

# Class table
story.append(Paragraph("Disease Counts:", subheading_style))
class_data = [
    ['Class', 'Count', '%'],
    ['No Finding', '60,361', '53.8%'],
    ['Infiltration', '19,894', '17.7%'],
    ['Effusion', '13,317', '11.9%'],
    ['Atelectasis', '11,559', '10.3%'],
    ['Nodule', '6,331', '5.7%'],
    ['Mass', '5,782', '5.2%'],
    ['Pneumothorax', '5,302', '4.7%'],
    ['Consolidation', '4,667', '4.2%'],
    ['Pleural_Thick.', '3,385', '3.0%'],
    ['Cardiomegaly', '2,776', '2.5%'],
    ['Emphysema', '2,516', '2.2%'],
    ['Edema', '2,303', '2.1%'],
    ['Fibrosis', '1,686', '1.5%'],
    ['Pneumonia', '1,431', '1.3%'],
    ['Hernia', '227', '0.2%'],
]
t = Table(class_data, colWidths=[2*inch, 1.5*inch, 1*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffe6e6')),  # No Finding highlighted
]))
story.append(t)
story.append(PageBreak())

# Class Imbalance
story.append(Paragraph("⚖️ Class Imbalance Analysis", heading_style))
story.append(Paragraph("<b>Imbalance Ratio:</b> 266:1 (No Finding vs Hernia)", body_style))
story.append(Paragraph("<b>Recommendation:</b> Use weighted loss or focal loss", body_style))
img_path = os.path.join(FIGURES_DIR, '02_class_imbalance_log.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=6.5*inch, height=3.5*inch))
story.append(Spacer(1, 0.3*inch))

# Multi-label Analysis
story.append(Paragraph("🏷️ Multi-Label Analysis", heading_style))
story.append(Paragraph("• <b>Mean labels per image:</b> 1.26", body_style))
story.append(Paragraph("• <b>Max labels:</b> 9 diseases in single image", body_style))
story.append(Paragraph("• <b>Single-label:</b> 91,324 (81.5%)", body_style))
story.append(Paragraph("• <b>Multi-label:</b> 20,796 (18.5%)", body_style))
img_path = os.path.join(FIGURES_DIR, '03_multilabel_distribution.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=3*inch))
story.append(PageBreak())

# Co-occurrence
story.append(Paragraph("🔗 Disease Co-occurrence", heading_style))
story.append(Paragraph("Top combinations: Infiltration+Effusion (4,000), Effusion+Atelectasis (3,275)", body_style))
img_path = os.path.join(FIGURES_DIR, '04_cooccurrence_matrix.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=6*inch, height=5*inch))
story.append(PageBreak())

# Demographics
story.append(Paragraph("👤 Patient Demographics", heading_style))
story.append(Paragraph("• <b>Age:</b> Mean 46.9 years (range 1-95)", body_style))
story.append(Paragraph("• <b>Gender:</b> Male 56.5%, Female 43.5%", body_style))
story.append(Paragraph("• <b>View:</b> PA 60%, AP 40%", body_style))
img_path = os.path.join(FIGURES_DIR, '05_demographics.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=5*inch))
story.append(PageBreak())

# Disease by Age
story.append(Paragraph("📊 Disease Prevalence by Age", heading_style))
img_path = os.path.join(FIGURES_DIR, '07_disease_by_age.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=4*inch))
story.append(Spacer(1, 0.3*inch))

# Disease by Gender
story.append(Paragraph("📊 Disease Prevalence by Gender", heading_style))
img_path = os.path.join(FIGURES_DIR, '06_disease_by_gender.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=3.5*inch))
story.append(PageBreak())

# BBox Analysis
story.append(Paragraph("📦 Bounding Box Analysis", heading_style))
story.append(Paragraph("<b>⚠️ Warning:</b> Only 984 bounding boxes (<1% coverage) - NOT suitable for YOLO!", body_style))
img_path = os.path.join(FIGURES_DIR, '09_bbox_analysis.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=3*inch))
story.append(Spacer(1, 0.3*inch))

# Train/Test Split
story.append(Paragraph("📁 Train/Test Split", heading_style))
img_path = os.path.join(FIGURES_DIR, '08_train_test_distribution.png')
if os.path.exists(img_path):
    story.append(RLImage(img_path, width=7*inch, height=3.5*inch))
story.append(PageBreak())

# Recommendations
story.append(Paragraph("💡 Key Insights & Recommendations", heading_style))
story.append(Spacer(1, 0.2*inch))

rec_data = [
    ['Finding', 'Recommendation'],
    ['Task Type', 'Multi-label Classification'],
    ['Loss Function', 'Weighted BCE or Focal Loss'],
    ['Output', 'Sigmoid (NOT Softmax)'],
    ['Metric', 'AUC-ROC'],
    ['Model', 'DenseNet-121 or ConvNeXt'],
    ['YOLO?', '❌ NO - Insufficient BBox data'],
]
t = Table(rec_data, colWidths=[2.5*inch, 4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ddd')),
    ('FONTSIZE', (0, 0), (-1, -1), 11),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fdf4')),
]))
story.append(t)

story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("<b>Suggested Models:</b>", subheading_style))
story.append(Paragraph("1. DenseNet-121 (CheXNet architecture - state-of-the-art)", body_style))
story.append(Paragraph("2. ConvNeXt (modern architecture)", body_style))
story.append(Paragraph("3. EfficientNet (good accuracy/efficiency)", body_style))
story.append(Paragraph("4. Vision Transformers (ViT, Swin)", body_style))

# Build PDF
print("Generating PDF...")
doc.build(story)
print(f"✅ PDF saved to: {PDF_FILE}")
