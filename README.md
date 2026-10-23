# 🧬 MedMash: Federated Oncology Diagnostics with Differential Privacy

MedMash is a decentralized, privacy-preserving machine learning infrastructure engineered for cross-institutional cancer diagnosis. It enables multiple clinical silos (AIIMS, Tata Memorial, Apollo) to collaboratively train a high-accuracy oncology classification model without centralizing sensitive patient records.

---

## 🏗️ System Architecture & Workflow

```text
       [ Central Server (Flower Orchestrator) ]
                     |          ^
     Global Model    |          | Aggregated Gradients
     Parameters (W)  |          | (FedAvg)
                     v          |
     +-----------------------------------------+
     |                    |                    |
     v                    v                    v
[ Silo 1: AIIMS ]   [ Silo 2: Tata ]    [ Silo 3: Apollo ]
- Local FNA Data    - Local FNA Data    - Local FNA Data
- PyTorch MLP       - PyTorch MLP       - PyTorch MLP
- Opacus DP-SGD     - Opacus DP-SGD     - Opacus DP-SGD
(ε=1.50, δ=1e-5)    (ε=1.50, δ=1e-5)    (ε=1.50, δ=1e-5)




Phase-by-Phase Breakdown:
Phase 1: Isolated Clinical Partitions: Partitioned fine needle aspirate (FNA) diagnostic metrics across simulated decentralized hospital silos. Zero cross-silo data egress.
Phase 2: Local Diagnostic Models: Configured PyTorch Multi-Layer Perceptrons (MLP) with BCEWithLogits loss and batch normalization.
Phase 3: Federated Orchestration: Orchestrated Federated Averaging (FedAvg) across 3 asynchronous silos via Flower (flwr), achieving 99.12% global aggregated accuracy.
Phase 4: Mathematical Privacy Shield (DP-SGD): Enforced Differential Privacy via Opacus with gradient clipping (C=1.0) and calibrated Gaussian noise addition, bounding privacy expenditure to ϵ≈1.50,δ=10 
−5
 .
Phase 5: Clinical Diagnostic Dashboard: Built an interactive diagnostic and network telemetry portal using Streamlit for real-time risk assessment and privacy audit tracking.
🔐 Differential Privacy Parameters
MedMash mathematically bounds data reconstruction and membership inference attacks via (ϵ,δ)-DP:
Privacy Budget (ϵ): ≤1.50 (Strict privacy limit)
Failure Probability (δ): 10 
−5
 
Max Gradient Norm (L 
2
​	
  Clip): 1.0
Mechanism: Gaussian Mechanism applied during local backpropagation.