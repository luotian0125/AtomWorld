#!/usr/bin/env python3
"""Generate passive-KMC first-passage targets for later Poisson calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.physics.keyframes import HCPKeyframeCriterion, KeyframeTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=(400, 600, 800)
    )
    parser.add_argument("--intervals", type=int, default=1000)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--maximum-microsteps", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=8111)
    parser.add_argument(
        "--max-conditional-mean-relative-error", type=float, default=0.15
    )
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/first_passage/passive_keyframe_intervals.json"),
    )
    return parser.parse_args()


def collect_intervals(env, count, maximum_microsteps, criterion=None):
    tracker = KeyframeTracker(criterion or HCPKeyframeCriterion())
    rows = []
    microsteps = 0
    interval_hops = Counter()
    starting_site_type = env.lattice.site_type_name(env.lattice.hydrogen_sites[0])
    while len(rows) < count and microsteps < maximum_microsteps:
        transition = env.step()
        microsteps += 1
        interval_hops[transition.hop_kind] += 1
        interval = tracker.observe(transition)
        if interval is not None:
            terminal_site_type = env.lattice.site_type_name(transition.target_site)
            rows.append(
                {
                    "starting_site_type": starting_site_type,
                    "terminal_site_type": terminal_site_type,
                    "microsteps": interval.microsteps,
                    "elapsed_physical_time_s": interval.elapsed_physical_time_s,
                    "terminal_event_id": interval.terminal_event_id,
                    "terminal_hop_kind": interval.terminal_hop_kind,
                    "reasons": list(interval.reasons),
                    "interval_hop_counts": dict(interval_hops),
                }
            )
            starting_site_type = terminal_site_type
            interval_hops = Counter()
    if len(rows) != count:
        raise RuntimeError(
            f"collected {len(rows)} of {count} requested keyframes before "
            f"the {maximum_microsteps}-microstep safety limit"
        )
    return rows


def summarize_intervals(rows):
    times = np.asarray(
        [row["elapsed_physical_time_s"] for row in rows], dtype=np.float64
    )
    microsteps = np.asarray([row["microsteps"] for row in rows], dtype=np.float64)
    terminal_counts = Counter(row["terminal_hop_kind"] for row in rows)
    starting_counts = Counter(row["starting_site_type"] for row in rows)
    mean_time = float(np.mean(times))
    return {
        "count": len(rows),
        "mean_elapsed_physical_time_s": mean_time,
        "median_elapsed_physical_time_s": float(np.median(times)),
        "p05_elapsed_physical_time_s": float(np.quantile(times, 0.05)),
        "p95_elapsed_physical_time_s": float(np.quantile(times, 0.95)),
        "time_cv_squared": (
            float(np.var(times, ddof=1) / (mean_time**2)) if len(times) > 1 else 0.0
        ),
        "mean_microsteps_per_keyframe": float(np.mean(microsteps)),
        "p95_microsteps_per_keyframe": float(np.quantile(microsteps, 0.95)),
        "terminal_hop_counts": dict(terminal_counts),
        "starting_site_type_counts": dict(starting_counts),
        "conditional_mean_microsteps": {
            site_type: float(
                np.mean(
                    [
                        row["microsteps"]
                        for row in rows
                        if row["starting_site_type"] == site_type
                    ]
                )
            )
            for site_type in sorted(starting_counts)
        },
        "conditional_mean_elapsed_time_s": {
            site_type: float(
                np.mean(
                    [
                        row["elapsed_physical_time_s"]
                        for row in rows
                        if row["starting_site_type"] == site_type
                    ]
                )
            )
            for site_type in sorted(starting_counts)
        },
    }


def validate_conditional_means(replica_rows, analytical, tolerance):
    empirical = {}
    relative_error = {}
    for site_type in ("T", "O"):
        empirical[site_type] = {
            "mean_microsteps": float(
                np.mean(
                    [
                        row["summary"]["conditional_mean_microsteps"][site_type]
                        for row in replica_rows
                    ]
                )
            ),
            "mean_elapsed_physical_time_s": float(
                np.mean(
                    [
                        row["summary"]["conditional_mean_elapsed_time_s"][site_type]
                        for row in replica_rows
                    ]
                )
            ),
        }
        relative_error[site_type] = {
            key: float(
                abs(empirical[site_type][key] - analytical[site_type][key])
                / analytical[site_type][key]
            )
            for key in ("mean_microsteps", "mean_elapsed_physical_time_s")
        }
    maximum_error = max(
        value
        for site_values in relative_error.values()
        for value in site_values.values()
    )
    return {
        "empirical_conditional_means": empirical,
        "relative_error": relative_error,
        "maximum_relative_error": float(maximum_error),
        "passed": bool(maximum_error <= tolerance),
    }


def main():
    args = parse_args()
    if args.intervals <= 0 or args.replicas <= 0 or args.maximum_microsteps <= 0:
        raise ValueError("intervals, replicas, and maximum-microsteps must be positive")
    if any(value <= 0.0 for value in args.temperatures):
        raise ValueError("temperatures must be positive")
    if args.max_conditional_mean_relative_error <= 0.0:
        raise ValueError("conditional-mean tolerance must be positive")

    results = []
    for temperature in args.temperatures:
        replica_rows = []
        for replica in range(args.replicas):
            env = HCPHydrogenKMC(
                size=(6, 6, 6),
                n_hydrogen=1,
                temperature_K=temperature,
                seed=args.seed + replica,
            )
            rows = collect_intervals(env, args.intervals, args.maximum_microsteps)
            replica_rows.append(
                {
                    "replica": replica,
                    "seed": args.seed + replica,
                    "summary": summarize_intervals(rows),
                    "intervals": rows,
                }
            )
        analytical = env.analytical_dilute_keyframe_first_passage()
        results.append(
            {
                "temperature_K": float(temperature),
                "analytical_conditional_means": analytical,
                "validation": validate_conditional_means(
                    replica_rows,
                    analytical,
                    args.max_conditional_mean_relative_error,
                ),
                "replicas": replica_rows,
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": "hcp_zrh_passive_kmc_keyframe_first_passage",
        "clock_owner": "passive_kmc",
        "policy_clock_validated": False,
        "keyframe_definition": {
            "slow_hop_kinds": ["TO", "OT", "OO"],
            "ordinary_TT_is_keyframe": False,
            "topology_changing_TT_is_keyframe": True,
        },
        "intervals_per_replica": args.intervals,
        "max_conditional_mean_relative_error": args.max_conditional_mean_relative_error,
        "results": results,
        "quality_gate": {
            "passed": bool(all(row["validation"]["passed"] for row in results))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for temperature_result in results:
        means = [
            row["summary"]["mean_microsteps_per_keyframe"]
            for row in temperature_result["replicas"]
        ]
        cv2 = [
            row["summary"]["time_cv_squared"] for row in temperature_result["replicas"]
        ]
        print(
            f"T={temperature_result['temperature_K']:7.1f} K "
            f"microsteps/keyframe={np.mean(means):.3f} "
            f"time CV^2={np.mean(cv2):.3f}"
        )
    print(f"wrote {args.output.resolve()}")
    if args.enforce and not payload["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
