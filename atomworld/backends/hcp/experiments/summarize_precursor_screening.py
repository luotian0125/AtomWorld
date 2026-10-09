#!/usr/bin/env python3
"""Summarize precursor screening as deltas from matched-seed dilute controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = (
    "pair_count:TTn2_attractive",
    "largest_attractive_cluster_fraction",
    "isolated_hydrogen_fraction",
    "T_fraction",
    "immediate_reverse_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/screening/precursor_sensitivity_stage1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/screening/precursor_sensitivity_stage1_summary.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/screening/precursor_sensitivity_stage1_summary.json"),
    )
    return parser.parse_args()


def _row_key(row):
    return float(row["temperature_K"]), int(row["n_hydrogen"])


def summarize(payload, metrics=DEFAULT_METRICS):
    replicas = int(payload["replicas"])
    if replicas <= 0:
        raise ValueError("replica count must be positive")
    dilute = {
        _row_key(row): row for row in payload["results"] if row["case"] == "dilute"
    }
    summary = []
    for row in payload["results"]:
        if row["case"] == "dilute":
            continue
        control = dilute[_row_key(row)]
        for metric in metrics:
            if metric not in row["metrics"] or metric not in control["metrics"]:
                continue
            treatment_stats = row["metrics"][metric]
            control_stats = control["metrics"][metric]
            treatment_mean = float(treatment_stats["mean"])
            control_mean = float(control_stats["mean"])
            delta = treatment_mean - control_mean
            paired_available = (
                len(row.get("replica_metrics", ())) == replicas
                and len(control.get("replica_metrics", ())) == replicas
                and all(metric in values for values in row["replica_metrics"])
                and all(metric in values for values in control["replica_metrics"])
            )
            if paired_available:
                paired_deltas = np.asarray(
                    [
                        row["replica_metrics"][index][metric]
                        - control["replica_metrics"][index][metric]
                        for index in range(replicas)
                    ],
                    dtype=np.float64,
                )
                standard_error = float(
                    np.std(paired_deltas, ddof=1) / np.sqrt(replicas)
                    if replicas > 1
                    else 0.0
                )
                standard_error_method = "matched_seed_paired_replica"
            else:
                standard_error = float(
                    np.sqrt(
                        (
                            float(treatment_stats["std_across_replicas"]) ** 2
                            + float(control_stats["std_across_replicas"]) ** 2
                        )
                        / replicas
                    )
                )
                standard_error_method = "conservative_independent_replica"
            summary.append(
                {
                    "temperature_K": float(row["temperature_K"]),
                    "n_hydrogen": int(row["n_hydrogen"]),
                    "H_atomic_fraction": float(row["H_atomic_fraction"]),
                    "case": row["case"],
                    "pair_energy_scale": float(row.get("pair_energy_scale", 1.0)),
                    "metric": metric,
                    "dilute_mean": control_mean,
                    "screening_mean": treatment_mean,
                    "delta_from_dilute": float(delta),
                    "standard_error": standard_error,
                    "standard_error_method": standard_error_method,
                    "absolute_delta_over_se": float(
                        abs(delta) / standard_error
                        if standard_error > 0.0
                        else np.inf if delta != 0.0 else 0.0
                    ),
                }
            )
    return summary


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("input screening quality gate did not pass")
    rows = summarize(payload)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_payload = {
        "schema_version": 1,
        "source_experiment": str(args.input.resolve()),
        "publication_ready_parameters": bool(
            payload.get("publication_ready_parameters", False)
        ),
        "uncertainty_note": (
            "Paired replica-level observations use matched-seed difference standard errors. "
            "Legacy results without replica rows use a conservative independent-replica SE."
        ),
        "rows": rows,
    }
    args.output_json.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output_csv.resolve()}")
    print(f"wrote {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
