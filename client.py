import argparse
from collections import OrderedDict
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from model import CancerDiagnosticMLP

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataloader(csv_path: str, batch_size: int = 16, shuffle: bool = True):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.float32).reshape(-1, 1)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def get_base_module(model):
    while hasattr(model, "_module"):
        model = model._module
    return model

def train_local_dp(model, dataloader, epochs=3, lr=0.005, target_epsilon=1.5, target_delta=1e-5):
    model.train()
    model = ModuleValidator.fix(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=1.0,
    )

    dp_model.train()
    for _ in range(epochs):
        for X_batch, y_batch in dp_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            dp_optimizer.zero_grad()
            preds = dp_model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            dp_optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return get_base_module(dp_model), epsilon

def evaluate_local(model, dataloader):
    raw_model = get_base_module(model)
    criterion = nn.BCEWithLogitsLoss()
    raw_model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = raw_model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(y_batch)
            probs = torch.sigmoid(preds)
            predicted_labels = (probs >= 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total

def get_parameters(net):
    raw_model = get_base_module(net)
    return [val.cpu().numpy() for _, val in raw_model.state_dict().items()]

def set_parameters(net, parameters):
    raw_model = get_base_module(net)
    params_dict = zip(raw_model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    raw_model.load_state_dict(state_dict, strict=True)

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_name: str):
        self.hospital_name = hospital_name
        self.train_loader = get_dataloader(f"data/{hospital_name}/train.csv", batch_size=16, shuffle=True)
        self.test_loader = get_dataloader(f"data/{hospital_name}/test.csv", batch_size=16, shuffle=False)

    def get_parameters(self, config):
        net = CancerDiagnosticMLP().to(DEVICE)
        return get_parameters(net)

    def fit(self, parameters, config):
        
        net = CancerDiagnosticMLP().to(DEVICE)
        set_parameters(net, parameters)
        trained_net, epsilon = train_local_dp(
            net,
            self.train_loader,
            epochs=3,
            lr=0.005,
            target_epsilon=1.5,
            target_delta=1e-5
        )
        print(f"[{self.hospital_name.upper()}] DP-SGD Round Complete | Privacy Spent (ε): {epsilon:.2f} (δ=1e-5)")
        return get_parameters(trained_net), len(self.train_loader.dataset), {"epsilon": float(epsilon)}

    def evaluate(self, parameters, config):
        net = CancerDiagnosticMLP().to(DEVICE)
        set_parameters(net, parameters)
        loss, accuracy = evaluate_local(net, self.test_loader)
        print(f"[{self.hospital_name.upper()}] Evaluation -> Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedMash Federated Client with Differential Privacy")
    parser.add_argument("--hospital", type=str, required=True, choices=["aiims", "tata", "apollo"], help="Hospital silo identifier")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Central server address")
    args = parser.parse_args()

    print(f"Securing and connecting {args.hospital.upper()} Silo to {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=HospitalClient(hospital_name=args.hospital)
    )