# Federated DP‑SGD on Adult Income

**Task.** Predict income > $50K. Data is partitioned across multiple “organizations” (clients). Raw data never leaves clients.

**Solution** Each client trains with **DP-SGD (Opacus)**: per-sample gradient clipping (C) + Gaussian noise (σ). We report **ε** per client for δ=1e-5.

**Goal.** Simulates enterprise data silos and non-IID distributions; evaluates utility vs privacy under federated averaging.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Prepare non-IID client shards (K=5 by default)
bash scripts/run_prepare.sh

# 2) Start server (terminal A)
bash scripts/run_server.sh

# 3) Start clients (terminal B)
bash scripts/run_clients.sh 5