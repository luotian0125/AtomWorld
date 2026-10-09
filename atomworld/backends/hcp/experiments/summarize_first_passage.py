#!/usr/bin/env python3
"""Aggregate passive first-passage intervals and test conditional exponentiality."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/first_passage/passive_keyframe_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/first_passage/passive_keyframe_summary.csv"),
    )
    parser.add_argument("--max-mean-relative-error", type=float, default=0.05)
    parser.add_argument("--max-cv2-deviation", type=float, default=0.10)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def aggregate(payload, max_mean_relative_error, max_cv2_deviation):
    rows = []
    for temperature_result in payload["results"]:
        analytical = temperature_result["analytical_conditional_means"]
        all_intervals = [
            interval
            for replica in temperature_result["replicas"]
            for interval in replica["intervals"]
        ]
        for site_type in ("T", "O"):
            selected = [
                interval
                for interval in all_intervals
                if interval["starting_site_type"] == site_type
            ]
            times = np.asarray(
                [interval["elapsed_physical_time_s"] for interval in selected],
                dtype=np.float64,
            )
            microsteps = np.asarray(
                [interval["microsteps"] for interval in selected], dtype=np.float64
            )
            mean_time = float(np.mean(times))
            mean_steps = float(np.mean(microsteps))
            cv2 = float(np.var(times, ddof=1) / mean_time**2)
            time_error = float(
                abs(mean_time - analytical[site_type]["mean_elapsed_physical_time_s"])
                / analytical[site_type]["mean_elapsed_physical_time_s"]
            )
            step_error = float(
                abs(mean_steps - analytical[site_type]["mean_microsteps"])
                / analytical[site_type]["mean_microsteps"]
            )
            checks = {
                "mean_time": time_error <= max_mean_relative_error,
                "mean_microsteps": step_error <= max_mean_relative_error,
                "conditional_exponentiality": abs(cv2 - 1.0) <= max_cv2_deviation,
            }
            rows.append(
                {
                    "temperature_K": float(temperature_result["temperature_K"]),
                    "starting_site_type": site_type,
                    "interval_count": len(selected),
                    "mean_elapsed_physical_time_s": mean_time,
                    "analytical_mean_elapsed_physical_time_s": float(
                        analytical[site_type]["mean_elapsed_physical_time_s"]
                    ),
                    "mean_time_relative_error": time_error,
                    "mean_microsteps": mean_steps,
                    "analytical_mean_microsteps": float(
                        analytical[site_type]["mean_microsteps"]
                    ),
                    "mean_microsteps_relative_error": step_error,
                    "conditional_time_cv_squared": cv2,
                    "quality_gate_passed": bool(all(checks.values())),
                }
            )
    return rows


def main():
    args = parse_args()
    if args.max_mean_relative_error <= 0.0 or args.max_cv2_deviation <= 0.0:
        raise ValueError("summary tolerances must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("clock_owner") != "passive_kmc":
        raise ValueError("first-passage input is not owned by passive KMC")
    rows = aggregate(
        payload,
        args.max_mean_relative_error,
        args.max_cv2_deviation,
    )
    quality_passed = bool(all(row["quality_gate_passed"] for row in rows))
    output = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "clock_owner": "passive_kmc",
        "policy_clock_validated": False,
        "interpretation": (
            "Conditional T/O first-passage times are tested against exponential CV^2=1; "
            "this validates calibration targets, not a learned-policy clock."
        ),
        "quality_gate": {"passed": quality_passed},
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"T={row['temperature_K']:7.1f} K start={row['starting_site_type']} "
            f"n={row['interval_count']:4d} CV2={row['conditional_time_cv_squared']:.4f} "
            f"time_error={row['mean_time_relative_error']:.4%} "
            f"gate={'PASS' if row['quality_gate_passed'] else 'FAIL'}"
        )
    print(f"wrote {args.output_json.resolve()}")
    print(f"wrote {args.output_csv.resolve()}")
    if args.enforce and not quality_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
