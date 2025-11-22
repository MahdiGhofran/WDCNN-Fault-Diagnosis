import torch
import torch.nn as nn

class WDCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(WDCNN, self).__init__()
        
        # Layer 1: Wide Convolution
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        # The output size calculation depends on input length (2048).
        # After L1: 2048 / 16 = 128 -> pool -> 64
        # After L2: 64 -> pool -> 32
        # After L3: 32 -> pool -> 16
        # After L4: 16 -> pool -> 8
        # After L5: 8 -> pool -> 4
        # Final features: 64 channels * 4 length = 256
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        # Expect input shape: (Batch, 1, 2048)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Test the model with a dummy input
    model = WDCNN(num_classes=10)
    dummy_input = torch.randn(32, 1, 2048)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

