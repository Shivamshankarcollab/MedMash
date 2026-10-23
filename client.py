import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from model import CancerDiagnosticMLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataloader(csv_path: str, batch_size: int = 16, shuffle: bool = True):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.float32).reshape(-1, 1)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_local(model, dataloader, epochs=5, lr=0.005):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

def evaluate_local(model, dataloader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(y_batch)
            probs = torch.sigmoid(preds)
            predicted_labels = (probs >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_name: str):
        self.hospital_name = hospital_name
        self.net = CancerDiagnosticMLP().to(DEVICE)
        self.train_loader = get_dataloader(f"data/{hospital_name}/train.csv", batch_size=16, shuffle=True)
        self.test_loader = get_dataloader(f"data/{hospital_name}/test.csv", batch_size=16, shuffle=False)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_local(self.net, self.train_loader, epochs=5, lr=0.005)
        return get_parameters(self.net), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = evaluate_local(self.net, self.test_loader)
        print(f"[{self.hospital_name.upper()}] Evaluation -> Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedMash Federated Client")
    parser.add_argument("--hospital", type=str, required=True, choices=["aiims", "tata", "apollo"], help="Hospital silo identifier")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Central server address")
    args = parser.parse_args()

    print(f"Connecting {args.hospital.upper()} Silo to Central Server at {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=HospitalClient(hospital_name=args.hospital)
    )