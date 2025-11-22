# WDCNN Fault Diagnosis

This repository contains a PyTorch implementation of the **WDCNN (Wide Deep Convolutional Neural Network)** for fault diagnosis of rotating machinery, based on the paper:
> *A New Convolutional Neural Network Based Data-Driven Fault Diagnosis Method* (Zhang et al., 2018).

## Project Overview
The goal is to classify bearing faults using raw vibration signals from the **Case Western Reserve University (CWRU)** dataset. The model achieves high accuracy by learning features directly from time-series data without manual feature extraction.

## Features
- **WDCNN Architecture**: Wide kernels in the first layer to capture high-frequency noise and low-frequency structural features.
- **End-to-End Learning**: Raw signal input -> Fault Classification.
- **Visualization**: Generates training curves and confusion matrices automatically.

## Requirements
- Python 3.8+
- PyTorch
- NumPy, SciPy
- Matplotlib, Seaborn, Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data
This script downloads sample `.mat` files from the CWRU website into the `data/` directory.
```bash
cd src
python download_data.py
```

### 2. Train the Model
Run the training script. This will train the model, evaluate it on a test set, and save the plots.
```bash
python train.py --epochs 20
```

### 3. Results
After training, check the `src/` directory for:
- `training_results.png`: Loss and Accuracy curves.
- `confusion_matrix.png`: Classification performance.
- `wdcnn_model.pth`: The saved model weights.

## Citation
If you use this code, please credit the original paper authors:
```
@article{zhang2018new,
  title={A new convolutional neural network based data-driven fault diagnosis method},
  author={Zhang, Wei and Peng, Gaoliang and Li, Chuanhao and Chen, Yuanhang and Zhang, Zhujun},
  journal={IEEE Transactions on Industrial Electronics},
  volume={65},
  number={5},
  pages={3993--4002},
  year={2018},
  publisher={IEEE}
}
```
