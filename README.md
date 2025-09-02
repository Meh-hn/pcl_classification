 3D Object Classification with PointNet, VoxNet, and ORION

This repository contains **PyTorch implementations** of several deep learning models for **3D object classification** on the **ModelNet10** and **ModelNet40** datasets.  
The project is structured to allow for easy experimentation and comparison between different architectures and data representations.

---

## 🚀 Features

- **Multiple Model Implementations**  
  - **PointNet**: Directly processes raw point cloud data.  
  - **VoxNet & ORION**: 3D CNNs operating on voxelized representations.  

- **Multi-Task Learning**  
  - ORION predicts both object class and orientation simultaneously.  

- **Reproducible Environment**  
  - Includes `environment.yml` to recreate the exact software environment with a single command.  

- **Configuration-Driven Experiments**  
  - Training hyperparameters are managed through `.yaml` config files, so no need to edit source code for new experiments.  

---


## 🔧 Setup & Installation

It is recommended to use **conda** for environment management.

### 1. Clone the Repository

```bash
git clone https://github.com/Meh-hn/pcl_classification.git
cd pcl_classification
```

2. Create and Activate the Conda Environment

This will install all dependencies (PyTorch, Trimesh, etc.) into an isolated environment named pcl_classification.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment (do this every time you work on the project)
conda activate pcl_classification
```

▶️ Usage

The workflow consists of three main stages: data preparation, model training, and evaluation.

1. Data Preparation

Download ModelNet10 and ModelNet40 datasets (in .off format) and place them under data/raw/.

```bash
python scripts/download_modelnet.py --dataset ModelNet10
```
Example expected structure:
data/raw/modelnet40/airplane/train/airplane_0001.off

Run preprocessing scripts:

For Point Cloud Data (PointNet)

Samples points from mesh surfaces and saves them as .npy files:

```bash
python scripts/pointnet_data_preparation.py --dataset modelnet10 --num_points 1024
```

For Voxelized Data (ORION/VoxNet)

Rotates, voxelizes, and pads meshes, saving them as .npy files:

```bash
python scripts/orion_data_preparation.py --dataset modelnet10 --grid_size 32
```

2. Model Training

Training runs are launched via runner scripts and config files.

Example: Train PointNet on ModelNet40

```bash
python scripts/train_pointnet.py --config configs/modelnet40/pointnet.yaml
```
Example: Train ORION on ModelNet10

```bash
python scripts/train_orion.py --config configs/modelnet10/orion_experiment.yaml
```
3. Evaluation



🏛️ Project Organization
This repo follows the Cookiecutter Data Science structure:



```bash
├── LICENSE          <- Open-source license.
├── Makefile         <- Convenience commands like `make data` or `make train`.
├── README.md        <- Project overview (you are here).
├── configs/         <- Experiment configs (.yaml).
├── data/
│   ├── raw/         <- Original datasets. (ignored by Git)
│   └── modelnet10/  <- Processed data (ignored by Git).
│   └── modelnet40/  <- Processed data (ignored by Git).
├── environment.yml  <- Conda environment definition.
├── notebooks/       <- Jupyter notebooks for experiments & visualization.
├── pyproject.toml   <- Package metadata.
├── scripts/         <- Data prep, training, evaluation scripts.
└── classification/  <- Core source code.
    ├── __init__.py
    ├── data_utils/  <- Dataset loaders.
    ├── engine/      <- Training & evaluation loops.
    ├── utils/       <- Losses, logging, point cloud transforms.
    └── models/      <- Model definitions (PointNet, ORION, etc.).
```

✅ TODO

 Implement PointNet++ for hierarchical feature learning.

 Add advanced data augmentation for point clouds.

 Experiment with loss weighting (gamma) for ORION.

## 📚 References

- **PointNet**: Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017).  
  *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.*  
  [Paper](https://arxiv.org/abs/1612.00593)

- **VoxNet**: Maturana, D., & Scherer, S. (2015).  
  *VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.*  
  [Paper](https://arxiv.org/abs/1505.06240)

- **ORION**: Sedaghat, N., Zolfaghari, M., Amiri, E., & Brox, T. (2016).  
  *Orientation-boosted Voxel Nets for 3D Object Recognition.*  
  [Paper](https://arxiv.org/abs/1604.03351)