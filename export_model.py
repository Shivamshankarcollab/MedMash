import torch
import pandas as pd
import numpy as np
from model import CancerDiagnosticMLP

DEVICE = torch.device("cpu")

def export_trained_model():
    
    dfs = [pd.read_csv(f"data/{h}/train.csv") for h in ["aiims", "tata", "apollo"]]
    df_all = pd.concat(dfs, ignore_index=True)
    
    X = torch.tensor(df_all.drop(columns=["target"]).values.astype(np.float32))
    y = torch.tensor(df_all["target"].values.astype(np.float32).reshape(-1, 1))

    model = CancerDiagnosticMLP()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "medmash_model.pt")
    print("Exported global federated model checkpoint to medmash_model.pt")

if __name__ == "__main__":
    export_trained_model()