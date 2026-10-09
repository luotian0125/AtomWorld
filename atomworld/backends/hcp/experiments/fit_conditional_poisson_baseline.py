#!/usr/bin/env python3
"""Fit and test a state-conditional exponential first-passage baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/first_passage/passive_keyframe_intervals.json"),
    )
    parser.add_argument("--train-replicas", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--test-replicas", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--max-test-mean-relative-error", type=float, default=0.10)
    parser.add_argument("--max-test-ks", type=float, default=0.08)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/first_passage/conditional_poisson_baseline.json"),
    )
    return parser.parse_args()


def _intervals_for_replicas(temperature_result, replica_ids):
    requested = set(map(int, replica_ids))
    return [
        interval
        for replica in temperature_result["replicas"]
        if int(replica["replica"]) in requested
        for interval in replica["intervals"]
    ]


def fit_exponential_rate(times):
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0 or np.any(times <= 0.0):
        raise ValueError("exponential samples must be non-empty and positive")
    return float(times.size / np.sum(times))


def exponential_nll(times, rate):
    times = np.asarray(times, dtype=np.float64)
    return float(np.mean(-np.log(rate) + rate * times))


def exponential_ks(times, rate):
    times = np.sort(np.asarray(times, dtype=np.float64))
    model_cdf = 1.0 - np.exp(-rate * times)
    upper_empirical = np.arange(1, len(times) + 1, dtype=np.float64) / len(times)
    lower_empirical = np.arange(0, len(times), dtype=np.float64) / len(times)
    return float(
        max(
            np.max(np.abs(upper_empirical - model_cdf)),
            np.max(np.abs(lower_empirical - model_cdf)),
        )
    )


def fit_temperature(temperature_result, train_replicas, test_replicas):
    train = _intervals_for_replicas(temperature_result, train_replicas)
    test = _intervals_for_replicas(temperature_result, test_replicas)
    global_rate = fit_exponential_rate(
        [row["elapsed_physical_time_s"] for row in train]
    )
    conditional_rates = {}
    state_results = {}
    conditional_test_nll_terms = []
    for site_type in ("T", "O"):
        train_times = [
            row["elapsed_physical_time_s"]
            for row in train
            if row["starting_site_type"] == site_type
        ]
        test_times = [
            row["elapsed_physical_time_s"]
            for row in test
            if row["starting_site_type"] == site_type
        ]
        rate = fit_exponential_rate(train_times)
        conditional_rates[site_type] = rate
        test_mean = float(np.mean(test_times))
        predicted_mean = float(1.0 / rate)
        mean_error = float(abs(test_mean - predicted_mean) / test_mean)
        ks = exponential_ks(test_times, rate)
        conditional_test_nll_terms.extend(
            [-np.log(rate) + rate * value for value in test_times]
        )
        state_results[site_type] = {
            "train_count": len(train_times),
            "test_count": len(test_times),
            "fitted_rate_per_second": rate,
            "predicted_mean_time_s": predicted_mean,
            "test_mean_time_s": test_mean,
            "test_mean_relative_error": mean_error,
            "test_ks": ks,
        }
    test_times_all = [row["elapsed_physical_time_s"] for row in test]
    conditional_nll = float(np.mean(conditional_test_nll_terms))
    global_nll = exponential_nll(test_times_all, global_rate)
    return {
        "temperature_K": float(temperature_result["temperature_K"]),
        "conditional_rates_per_second": conditional_rates,
        "global_rate_per_second": global_rate,
        "state_results": state_results,
        "test_nll": {
            "conditional": conditional_nll,
            "global": global_nll,
            "global_minus_conditional": float(global_nll - conditional_nll),
        },
    }


def main():
    args = parse_args()
    if set(args.train_replicas) & set(args.test_replicas):
        raise ValueError("train and test replica IDs must be disjoint")
    if args.max_test_mean_relative_error <= 0.0 or args.max_test_ks <= 0.0:
        raise ValueError("Poisson validation tolerances must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("clock_owner") != "passive_kmc":
        raise ValueError("Poisson baseline requires passive-KMC first-passage data")
    results = [
        fit_temperature(row, args.train_replicas, args.test_replicas)
        for row in payload["results"]
    ]
    for row in results:
        checks = {
            "held_out_mean": max(
                values["test_mean_relative_error"]
                for values in row["state_results"].values()
            )
            <= args.max_test_mean_relative_error,
            "held_out_ks": max(
                values["test_ks"] for values in row["state_results"].values()
            )
            <= args.max_test_ks,
            "conditioning_not_worse": row["test_nll"]["global_minus_conditional"]
            >= -0.005,
        }
        row["quality_gate"] = {"passed": bool(all(checks.values())), "checks": checks}
    quality_passed = bool(all(row["quality_gate"]["passed"] for row in results))
    output = {
        "schema_version": 1,
        "experiment": "hcp_zrh_conditional_poisson_baseline",
        "source": str(args.input.resolve()),
        "train_replicas": args.train_replicas,
        "test_replicas": args.test_replicas,
        "policy_clock_validated": False,
        "scope": "held-out passive-KMC baseline only",
        "results": results,
        "quality_gate": {"passed": quality_passed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in results:
        print(
            f"T={row['temperature_K']:7.1f} K "
            f"mean_error_T={row['state_results']['T']['test_mean_relative_error']:.4%} "
            f"mean_error_O={row['state_results']['O']['test_mean_relative_error']:.4%} "
            f"KSmax={max(x['test_ks'] for x in row['state_results'].values()):.4f} "
            f"dNLL={row['test_nll']['global_minus_conditional']:.6f} "
            f"gate={'PASS' if row['quality_gate']['passed'] else 'FAIL'}"
        )
    print(f"wrote {args.output.resolve()}")
    if args.enforce and not quality_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
