import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os
import re

class CWRUDataset(Dataset):
    def __init__(self, root_dir, signal_length=2048, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .mat files.
            signal_length (int): Length of the signal segments.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.signal_length = signal_length
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Mapping filenames to Class IDs
        # 0: Normal
        # 1: Inner Race (0.007")
        # 2: Ball (0.007")
        # 3: Outer Race (0.007", Centered)
        self.file_map = {
            "97.mat": 0,
            "105.mat": 1,
            "118.mat": 2,
            "130.mat": 3
        }
        
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.root_dir}...")
        
        for filename, label in self.file_map.items():
            file_path = os.path.join(self.root_dir, filename)
            if not os.path.exists(file_path):
                print(f"Warning: {filename} not found in {self.root_dir}. Skipping.")
                continue
                
            try:
                mat = sio.loadmat(file_path)
                
                # Find the Drive End (DE) data key
                # Pattern: X + 3 digits + _DE_time
                # e.g., X097_DE_time
                de_key = None
                for key in mat.keys():
                    if key.endswith("_DE_time"):
                        de_key = key
                        break
                
                if de_key is None:
                    print(f"Error: Could not find DE_time key in {filename}")
                    continue
                    
                signal = mat[de_key].flatten()
                n_samples = len(signal)
                n_segments = n_samples // self.signal_length
                
                print(f"Loaded {filename}: {n_samples} points -> {n_segments} segments (Class {label})")
                
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
        print(f"Total loaded: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Reshape to (1, signal_length) for Conv1d input (Channels, Length)
        sample = sample.reshape(1, -1)
        sample = torch.from_numpy(sample).float()
        
        # Convert label to tensor
        label = torch.tensor(label).long()
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
