import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import WDCNN
from dataset import CWRUDataset
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    num_classes = 10 # Updated to 10 classes

    # Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), 'data')
    
    full_dataset = CWRUDataset(root_dir=data_path)
    
    if len(full_dataset) == 0:
        print("No data loaded!")
        return

    # Split: Train (70%), Val (15%), Test (15%)
    total_count = len(full_dataset)
    train_size = int(0.7 * total_count)
    val_size = int(0.15 * total_count)
    test_size = total_count - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Model
    model = WDCNN(num_classes=num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # WDCNN paper uses SGD, but Adam is faster for quick results

    # History
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)

            # Add minimal random noise during training to prevent absolute overfitting
            # signal_noise = signals + 0.01 * torch.randn_like(signals) 
            
            outputs = model(signals)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'wdcnn_model_best.pth')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    print("Training finished. Loading best model for testing...")
    model.load_state_dict(torch.load('wdcnn_model_best.pth'))
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig('training_results.png')
    
    # Final Test with Confusion Matrix
    print("Evaluating on strictly held-out Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_test_acc = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
            
    cm = confusion_matrix(all_labels, all_preds)
    # Short names for 10 classes
    class_names = ['Norm', 'IR07', 'B07', 'OR07', 'IR14', 'B14', 'OR14', 'IR21', 'B21', 'OR21']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (10 Classes) - Acc: {final_test_acc:.1f}%')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30) # Increased epochs for harder problem
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    train_and_evaluate(args)
