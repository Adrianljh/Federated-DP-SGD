from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
import flwr as fl
from src.model import MLP
from src.utils import ensure_dir, set_seed, device_auto, save_json

class DPClient(fl.client.NumPyClient):
    def __init__(self, client_dir: Path, batch_size=512, lr=1e-3, clip=1.0, noise_multiplier=None, target_epsilon=None, delta=1e-5, local_epochs=1, seed=42):
        set_seed(seed)
        self.client_dir = Path(client_dir)
        self.device = device_auto()
        # load data
        Xtr = np.load(self.client_dir / 'X_train.npy'); ytr = np.load(self.client_dir / 'y_train.npy')
        Xte = np.load(self.client_dir / 'X_test.npy');  yte = np.load(self.client_dir / 'y_test.npy')
        self.d_in = Xtr.shape[1]
        self.train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
        self.test_ds  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32))
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.local_epochs = local_epochs
        self.checkpoint_dir = Path('outputs/clients') / self.client_dir.name
        ensure_dir(self.checkpoint_dir)

    def _new_model(self):
        m = MLP(self.d_in).to(self.device)
        return m

    def get_parameters(self, config):
        # initial parameters for round 0
        return [val.cpu().numpy() for _, val in self._new_model().state_dict().items()]

    def set_parameters(self, model, parameters):
        sd = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        model = self._new_model()
        self.set_parameters(model, parameters)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.BCEWithLogitsLoss()

        # DP wrapping
        pe = PrivacyEngine()
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        if self.target_epsilon is None:
            model, opt, train_loader = pe.make_private(
                module=model, optimizer=opt, data_loader=train_loader,
                noise_multiplier=self.noise_multiplier, max_grad_norm=self.clip,
            )
        else:
            model, opt, train_loader = pe.make_private_with_epsilon(
                module=model, optimizer=opt, data_loader=train_loader,
                target_epsilon=self.target_epsilon, target_delta=self.delta,
                epochs=int(config.get('local_epochs', self.local_epochs)), max_grad_norm=self.clip,
            )

        model.train()
        for _ in range(int(config.get('local_epochs', self.local_epochs))):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); logits = model(xb); loss = crit(logits, yb)
                loss.backward(); opt.step()

        # epsilon achieved this round (cumulative per training session)
        eps = pe.get_epsilon(self.delta)
        save_json({"epsilon": float(eps), "delta": float(self.delta)}, self.checkpoint_dir / 'epsilon.json')
        torch.save(model.state_dict(), self.checkpoint_dir / 'best.pt')

        new_params = [v.cpu().numpy() for _, v in model.state_dict().items()]
        num_examples = len(self.train_ds)
        metrics = {"epsilon": float(eps)}
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        from src.metrics import compute
        model = self._new_model()
        self.set_parameters(model, parameters)
        model.eval()
        with torch.no_grad():
            X, y = self.test_ds.tensors
            logits = model(X.to(self.device)).cpu().numpy()
            probs = 1/(1+np.exp(-logits))
            y_np = y.numpy().astype(int)
            metrics = compute(y_np, probs)
            loss = float(((y - torch.tensor(probs)).abs()).mean().item())
        return loss, len(self.test_ds), metrics


def start_client(cid: int, dp_args: dict):
    client_dir = Path(f"data/clients/client_{cid}")
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=DPClient(client_dir, **dp_args))