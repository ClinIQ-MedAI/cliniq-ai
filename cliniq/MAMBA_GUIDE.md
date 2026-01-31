# Mamba Quick Reference for ClinIQ

Mamba is a fast, drop-in replacement for conda. It's **3-5x faster** for environment creation and package installation.

## Installation

```bash
# Install mamba in base environment (one-time setup)
conda install -n base -c conda-forge mamba
```

## Common Commands

### Environment Creation
```bash
# Create environment from file (FAST!)
mamba env create -f environment.yml

# Create environment with specific packages
mamba create -n myenv python=3.10 pytorch torchvision
```

### Environment Management
```bash
# Activate (use conda, not mamba)
conda activate cliniq

# Deactivate
conda deactivate

# List environments
conda env list
# or
mamba env list

# Remove environment
conda env remove -n cliniq
```

### Package Management
```bash
# Install packages (FAST!)
mamba install pytorch torchvision -c pytorch

# Update packages
mamba update --all

# Update specific package
mamba update pytorch

# Remove package
mamba remove package_name
```

### Environment Updates
```bash
# Update environment from file (FAST!)
mamba env update -f environment.yml --prune

# Export environment
conda env export > environment_backup.yml
```

## Mamba vs Conda

| Task | Conda Command | Mamba Command | Speed Gain |
|------|--------------|---------------|------------|
| Create env | `conda env create -f env.yml` | `mamba env create -f env.yml` | **3-5x faster** |
| Install packages | `conda install package` | `mamba install package` | **3-5x faster** |
| Update env | `conda env update -f env.yml` | `mamba env update -f env.yml` | **3-5x faster** |
| Activate env | `conda activate env` | `conda activate env` | Same |
| List envs | `conda env list` | Both work | Same |

## ClinIQ Workflow

```bash
# Initial setup (one-time)
cd /home/moabouag/far/cliniq
conda install -n base -c conda-forge mamba
mamba env create -f environment.yml

# Daily workflow
conda activate cliniq
cd Oral-Dental/oral-xray
python train.py --config configs/classification_resnet50.yaml --exp-name exp1

# When environment.yml changes
cd /home/moabouag/far/cliniq
mamba env update -f environment.yml --prune
```

## Tips

1. **Use mamba for installation/updates** - It's much faster
2. **Use conda for activation** - Standard practice
3. **Both share the same environments** - They're interchangeable
4. **Mamba uses less memory** - Better for large environments

## Troubleshooting

**Mamba not found?**
```bash
# Install mamba
conda install -n base -c conda-forge mamba

# Verify installation
mamba --version
```

**Conflicts during installation?**
```bash
# Mamba is better at resolving conflicts, but if issues persist:
# 1. Clear cache
mamba clean --all

# 2. Recreate environment
conda env remove -n cliniq
mamba env create -f environment.yml
```

**Switch back to conda?**
```bash
# Just replace 'mamba' with 'conda' in any command
conda env create -f environment.yml
```

---

**Mamba = Conda but faster! 🐍⚡**
