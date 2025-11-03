from __future__ import annotations
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir, set_seed

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES = {"train": "adult.data", "test": "adult.test"}
RAW = Path("data/raw"); CLIENTS_DIR = Path("data/clients")
COLUMNS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']


def download():
    ensure_dir(RAW)
    for fname in FILES.values():
        url = UCI_BASE + fname
        dst = RAW / fname
        if not dst.exists():
            print(f"Downloading {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
        else:
            print(f"Found {dst}")


def load_clean() -> pd.DataFrame:
    train_path = RAW / FILES['train']
    test_path  = RAW / FILES['test']
    df_tr = pd.read_csv(train_path, header=None, names=COLUMNS, na_values=' ?', skipinitialspace=True)
    df_te = pd.read_csv(test_path,  header=0,   names=COLUMNS, na_values=' ?', skipinitialspace=True)
    df_te['income'] = df_te['income'].str.replace('.', '', regex=False)
    df = pd.concat([df_tr, df_te], ignore_index=True).dropna().reset_index(drop=True)
    return df


def encode(df: pd.DataFrame):
    # Minimal encoding: one-hot categoricals; scale numerics
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    X_cols = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
    y = (df['income'].str.contains('>50K')).astype(int).values
    X_df = df[X_cols].copy()
    cat = X_df.select_dtypes(include=['object']).columns.tolist()
    num = [c for c in X_df.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
    ])
    X_all = pre.fit_transform(X_df)
    return X_all.astype(np.float32), y.astype(np.int64)


def partition_non_iid(X: np.ndarray, y: np.ndarray, K: int = 5, seed: int = 42):
    set_seed(seed)
    idx = np.arange(len(y))
    pos, neg = idx[y == 1], idx[y == 0]
    np.random.shuffle(pos); np.random.shuffle(neg)

    props = np.array([0.18, 0.16, 0.20, 0.24, 0.22])
    pos_splits = np.split(pos, (props.cumsum()[:-1] * len(pos)).astype(int))
    neg_splits = np.split(neg, (props.cumsum()[:-1] * len(neg)).astype(int))

    clients = []
    for j in range(K):
        sel = np.concatenate([pos_splits[j], neg_splits[j]])
        np.random.shuffle(sel)
        tr, te = train_test_split(sel, test_size=0.2, stratify=y[sel], random_state=seed)
        clients.append({"train": tr, "test": te})
    return clients


def save_clients(X, y, parts):
    for j, part in enumerate(parts):
        base = CLIENTS_DIR / f"client_{j+1}"
        ensure_dir(base)
        np.save(base / 'X_train.npy', X[part['train']])
        np.save(base / 'y_train.npy', y[part['train']])
        np.save(base / 'X_test.npy',  X[part['test']])
        np.save(base / 'y_test.npy',  y[part['test']])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--clients', type=int, default=5)
    args = p.parse_args()
    download(); df = load_clean(); X, y = encode(df)
    parts = partition_non_iid(X, y, K=args.clients)
    save_clients(X, y, parts)
    print(f"Saved {args.clients} client shards under {CLIENTS_DIR}")