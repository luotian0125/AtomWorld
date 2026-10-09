#!/usr/bin/env python3
"""Generate the analytical HCP interstitial-event spectrum used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.lattices.hcp_interstitial import HOP_KINDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=(300, 400, 500, 600, 800, 1000),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/physics_validation/hcp_event_space_spectrum.json"),
    )
    return parser.parse_args()


def characterize_temperature(temperature):
    env = HCPHydrogenKMC(size=(3, 3, 3), temperature_K=temperature, seed=1)
    rates = {kind: env.parameters.rate(kind, temperature) for kind in HOP_KINDS}
    fastest = max(rates, key=rates.get)
    slowest = min(rates, key=rates.get)
    fractions = env.analytical_hop_fractions()
    return {
        "temperature_K": float(temperature),
        "rates_per_second": rates,
        "analytical_hop_fractions": fractions,
        "analytical_site_fractions": env.analytical_site_fractions(),
        "fastest_event": fastest,
        "slowest_event": slowest,
        "fastest_over_slowest_rate": float(rates[fastest] / rates[slowest]),
        "TT_over_TO_rate": float(rates["TT"] / rates["TO"]),
        "TO_OT_flux_balance_residual": float(abs(fractions["TO"] - fractions["OT"])),
        "analytical_diffusivity_m2_per_s": env.analytical_dilute_diffusivity_m2_per_s(),
    }


def main():
    args = parse_args()
    if any(temperature <= 0.0 for temperature in args.temperatures):
        raise ValueError("temperatures must be positive")
    results = [characterize_temperature(value) for value in args.temperatures]
    payload = {
        "schema_version": 1,
        "experiment": "hcp_zrh_analytical_event_space_spectrum",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in results:
        print(
            f"T={row['temperature_K']:7.1f} K "
            f"fast/slow={row['fastest_over_slowest_rate']:.3e} "
            f"TT/TO={row['TT_over_TO_rate']:.3e} "
            f"TT fraction={row['analytical_hop_fractions']['TT']:.6f}"
        )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
