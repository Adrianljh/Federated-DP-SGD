from __future__ import annotations
from pathlib import Path
import flwr as fl
import numpy as np
from src.utils import append_jsonl, ensure_dir

LOG = Path("outputs/server/round_metrics.jsonl")

class LogFederatedStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        eps_list = []
        for _, fit_res in results:
            eps = fit_res.metrics.get("epsilon") if fit_res.metrics else None
            if eps is not None:
                eps_list.append(float(eps))
        if eps_list:
            append_jsonl({"round": rnd, "client_epsilons": eps_list}, LOG)
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        if results:
            metrics_list = [(res.num_examples, res.metrics) for _, res in results]
            agg_metrics = _agg_eval_metrics(metrics_list)
            payload = {"round": rnd, **agg_metrics}
            append_jsonl(payload, LOG)
        return super().aggregate_evaluate(rnd, results, failures)


def _agg_eval_metrics(metrics_list):
    """
    metrics_list: List[Tuple[int, Dict[str, float]]]
    e.g. [(num_examples_1, {'accuracy': 0.8, 'f1': 0.7}), ...]
    Computes weighted mean by client size.
    """
    def wmean(key: str):
        pairs = [
            (n, float(m[key]))
            for (n, m) in metrics_list
            if isinstance(m, dict) and key in m and m[key] is not None
        ]
        if not pairs:
            return None
        total_n = sum(n for n, _ in pairs)
        return float(sum(n * v for n, v in pairs) / total_n)

    return {
        "accuracy": wmean("accuracy"),
        "f1": wmean("f1"),
        "roc_auc": wmean("roc_auc"),
    }


def start_server(rounds=30, min_fit_clients=2, min_available_clients=2):
    ensure_dir(LOG.parent)
    strategy = LogFederatedStrategy(
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=_agg_eval_metrics,  # proper signature
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--min_fit_clients", type=int, default=2)
    p.add_argument("--min_available_clients", type=int, default=2)
    args = p.parse_args()
    start_server(**vars(args))
