import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 1. Diagnostic Neural Network (MLP)
class CancerMLP(nn.Module):
    def __init__(self, input_dim=30, hidden1=64, hidden2=32, num_classes=2):
        super(CancerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 2. PyTorch Dataset Contract for Hospital Silos
class CancerDataset(Dataset):
    def __init__(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        feature_cols = [c for c in df.columns if c != "target"]
        
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(csv_file_path, batch_size=16, shuffle=True):
    dataset = CancerDataset(csv_file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 3. Local Training Loop
def train(net, train_loader, epochs=5, lr=0.01, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    net.train()

    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss = epoch_loss / len(train_loader)
        
    return total_loss

# 4. Local Evaluation Loop
def evaluate(net, test_loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# 5. Standalone Sanity Verification
if __name__ == "__main__":
    print("Running local training sanity check on AIIMS silo...")
    train_loader = get_dataloader("data/aiims/train.csv", batch_size=16, shuffle=True)
    test_loader = get_dataloader("data/aiims/test.csv", batch_size=16, shuffle=False)

    model = CancerMLP()
    initial_loss, initial_acc = evaluate(model, test_loader)
    print(f"Untrained AIIMS Model -> Loss: {initial_loss:.4f}, Accuracy: {initial_acc * 100:.2f}%")

    final_loss = train(model, train_loader, epochs=20, lr=0.005)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Trained AIIMS Model (20 Epochs) -> Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")
