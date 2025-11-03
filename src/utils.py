import json, random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | Path):
    path = Path(path); ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def append_jsonl(obj, path: str | Path):
    path = Path(path); ensure_dir(path.parent)
    with open(path, 'a') as f:
        f.write(json.dumps(obj) + "\n")


def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')