# WDCNN Fault Diagnosis (10 Classes)

This repository contains a PyTorch implementation of the **WDCNN (Wide Deep Convolutional Neural Network)** for fault diagnosis of rotating machinery, based on the paper:
> *A New Convolutional Neural Network Based Data-Driven Fault Diagnosis Method* (Zhang et al., 2018).

## Project Overview
The model classifies **10 different bearing fault conditions** (Normal + 3 Fault Types × 3 Severities) using raw vibration signals from the **Case Western Reserve University (CWRU)** dataset.

## Dataset Classes
1. Normal
2. Inner Race (0.007")
3. Ball (0.007")
4. Outer Race (0.007")
5. Inner Race (0.014")
6. Ball (0.014")
7. Outer Race (0.014")
8. Inner Race (0.021")
9. Ball (0.021")
10. Outer Race (0.021")

## Requirements
- Python 3.8+
- PyTorch, NumPy, SciPy, Matplotlib, Seaborn, Scikit-learn

```bash
pip install -r requirements.txt
```

## Usage
1. **Download Data**: `python src/download_data.py` (Downloads ~10 files)
2. **Train**: `python src/train.py --epochs 30`
3. **Results**: Check `src/confusion_matrix.png` for performance visualization.

## Scientific Validity
- Data is strictly split into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
- The Test set is held out until the final evaluation to prevent data leakage.
