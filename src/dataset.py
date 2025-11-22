import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
import re

class CWRUDataset(Dataset):
    def __init__(self, root_dir, signal_length=2048, transform=None):
        self.root_dir = root_dir
        self.signal_length = signal_length
        self.transform = transform
        self.data = []
        self.labels = []
        
        # 10-Class Problem Definition
        self.file_map = {
            "97.mat": 0,   # Normal
            "105.mat": 1,  # IR 007
            "118.mat": 2,  # Ball 007
            "130.mat": 3,  # OR 007
            "169.mat": 4,  # IR 014
            "185.mat": 5,  # Ball 014
            "197.mat": 6,  # OR 014
            "209.mat": 7,  # IR 021
            "222.mat": 8,  # Ball 021
            "234.mat": 9   # OR 021
        }
        
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.root_dir}...")
        
        for filename, label in self.file_map.items():
            file_path = os.path.join(self.root_dir, filename)
            if not os.path.exists(file_path):
                print(f"Warning: {filename} not found. Skipping.")
                continue
                
            try:
                mat = sio.loadmat(file_path)
                
                # Find DE_time key
                de_key = None
                for key in mat.keys():
                    if key.endswith("_DE_time"):
                        de_key = key
                        break
                
                if de_key is None:
                    print(f"Error: No DE_time in {filename}")
                    continue
                    
                signal = mat[de_key].flatten()
                n_samples = len(signal)
                n_segments = n_samples // self.signal_length
                
                # Limit samples per class to avoid extreme imbalance (optional)
                # For now, we take all available segments
                
                for i in range(n_segments):
                    start = i * self.signal_length
                    end = start + self.signal_length
                    segment = signal[start:end]
                    self.data.append(segment)
                    self.labels.append(label)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Total loaded: {len(self.data)} samples across 10 classes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        sample = sample.reshape(1, -1)
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label).long()
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
