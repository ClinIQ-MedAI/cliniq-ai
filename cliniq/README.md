# ClinIQ - Medical Imaging AI Platform

Unified environment for all ClinIQ medical imaging projects:
- **DMRI**: Diffusion MRI processing and tractography
- **Oral-Dental**: Dental X-ray classification, detection, and segmentation
- **Future projects**: Additional medical imaging modalities

## 🚀 Quick Setup

### Create Environment

**Recommended: Use mamba (much faster)**
```bash
cd /home/moabouag/far/cliniq

# Install mamba (if not already installed)
conda install -n base -c conda-forge mamba

# Create environment with mamba (takes 2-5 min instead of 10-15)
mamba env create -f environment.yml

# Activate
conda activate cliniq

# Verify installation
python -c "import torch, dipy, cv2; print('✓ All packages loaded successfully')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Environment Management

```bash
# Update environment after changes to environment.yml
mamba env update -f environment.yml --prune
# Or: conda env update -f environment.yml --prune

# Deactivate
conda deactivate

# Remove environment
conda env remove -n cliniq

# Export current environment
conda env export > environment_backup.yml
```

## 📁 Project Structure

```
cliniq/
├── environment.yml          # Unified conda environment (YOU ARE HERE)
├── requirements.txt         # Backup pip requirements
├── DMRI/                    # Diffusion MRI pipeline
│   ├── src/
│   ├── notebooks/
│   └── README.md
└── Oral-Dental/             # Dental imaging
    ├── oral-classify/
    ├── oral-detect/
    └── oral-xray/           # Main X-ray ML pipeline
```

## 🔧 Working with Specific Projects

### Oral X-Ray Project

```bash
conda activate cliniq
cd Oral-Dental/oral-xray

# Follow the quickstart guide
cat QUICKSTART.md

# Train a model
python train.py --config configs/classification_resnet50.yaml --exp-name exp1
```

### DMRI Project

```bash
conda activate cliniq
cd DMRI

# Run DMRI processing
# (See DMRI/README.md for specific instructions)
```

## 📦 Included Packages

### Deep Learning & ML
- PyTorch 2.0+ with CUDA support
- torchvision, timm (PyTorch Image Models)
- ultralytics (YOLOv8)
- scikit-learn

### Medical Imaging
- **DIPY**: Diffusion MRI processing
- **nibabel**: NIfTI file handling
- **pycocotools**: Medical image annotations

### Computer Vision
- OpenCV
- Albumentations (augmentation)
- Pillow

### Visualization
- Matplotlib, Seaborn
- FURY (3D tractography)
- pygfx, wgpu (modern rendering)
- Jupyter Lab

### MLOps & Production
- MLflow, TensorBoard, W&B (experiment tracking)
- DVC (data versioning)
- FastAPI, uvicorn (API deployment)
- ONNX, ONNX Runtime (model export)

### Development
- pytest, black, flake8
- Jupyter Lab

## 🖥️ CUDA Version

The environment is configured for **CUDA 11.8**. If you have a different CUDA version:

```bash
# Check your CUDA version
nvidia-smi

# Edit environment.yml and change:
- cudatoolkit=11.8  # Change to 11.7, 12.1, etc.

# Or for CPU-only (no GPU):
# Remove cudatoolkit line and change:
# - pytorch>=2.0.0
# to:
# - pytorch-cpu>=2.0.0
```

## 📝 Additional Setup

### For Oral X-Ray Dataset Download

```bash
# Option 1: Install rclone (no sudo required)
cd ~
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
cd rclone-*-linux-amd64

# Copy to user bin directory
mkdir -p ~/.local/bin
cp rclone ~/.local/bin/

# Add to PATH in ~/.bashrc if not already there
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Configure for Google Drive
rclone config  # Follow prompts to setup Google Drive

# Option 2: Install gdown (simpler, but slower for large datasets)
pip install gdown
```

### For Jupyter Notebooks

```bash
conda activate cliniq

# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Update environment (use mamba for speed)
mamba env update -f environment.yml --prune

# Or recreate from scratch
conda env remove -n cliniq
mamba env create -f environment.yml
```

### CUDA Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, check:
# 1. nvidia-smi (driver installed?)
# 2. cudatoolkit version matches your CUDA
# 3. Reinstall PyTorch with correct CUDA version
```

### Package Conflicts

```bash
# Create fresh environment
conda create -n cliniq-test python=3.10
conda activate cliniq-test

# Install packages incrementally
conda install pytorch torchvision -c pytorch
# ... then add others
```

## 📚 Project Documentation

- [DMRI README](DMRI/README.md) - Diffusion MRI pipeline
- [Oral X-Ray README](Oral-Dental/oral-xray/README.md) - Dental X-ray ML
- [Oral X-Ray Quick Start](Oral-Dental/oral-xray/QUICKSTART.md) - 15-min setup

## 🤝 Contributing

This is a unified environment for all ClinIQ projects. When adding new dependencies:

1. Add to `environment.yml` (for conda packages)
2. Or add to `pip:` section (for pip-only packages)
3. Test installation: `conda env update -f environment.yml`
4. Document in this README

## 📧 Support

For environment issues, check the troubleshooting section above or refer to specific project READMEs.

---

**One environment for all medical imaging AI projects! 🏥🤖**
