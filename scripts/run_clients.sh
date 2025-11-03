#!/usr/bin/env bash
set -euo pipefail
K=${1:-5}
for i in $(seq 1 $K); do
  python - <<PY &
from src.client_dp import start_client
start_client(${i}, dp_args={
  'batch_size':1024,
  'lr':1e-3,
  'clip':1.0,
  'noise_multiplier':0.8,
  'target_epsilon': None,
  'delta':1e-5,
  'local_epochs':3,
  'seed': 42+${i}
})
PY
  sleep 0.5
done
wait