"""Publication-traceable parameter loading for dilute H diffusion in alpha-Zr."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


REQUIRED_HOPS = ("TT", "TO", "OT", "OO")


@dataclass(frozen=True)
class ZrHKMCParameters:
    name: str
    site_energies_eV: dict[str, float]
    migration_barriers_eV: dict[str, float]
    attempt_frequencies_Hz: dict[str, float]
    lattice_a_angstrom: float
    lattice_c_angstrom: float
    source: str
    publication_ready: bool

    def validate(self):
        if set(self.site_energies_eV) != {"T", "O"}:
            raise ValueError("site_energies_eV must contain exactly T and O")
        if set(self.migration_barriers_eV) != set(REQUIRED_HOPS):
            raise ValueError(f"migration barriers must contain {REQUIRED_HOPS}")
        if set(self.attempt_frequencies_Hz) != set(REQUIRED_HOPS):
            raise ValueError(f"attempt frequencies must contain {REQUIRED_HOPS}")
        numeric = [
            *self.site_energies_eV.values(),
            *self.migration_barriers_eV.values(),
            *self.attempt_frequencies_Hz.values(),
            self.lattice_a_angstrom,
            self.lattice_c_angstrom,
        ]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Zr-H parameters must be finite")
        if any(value <= 0.0 for value in self.migration_barriers_eV.values()):
            raise ValueError("migration barriers must be positive")
        if any(value <= 0.0 for value in self.attempt_frequencies_Hz.values()):
            raise ValueError("attempt frequencies must be positive")
        if self.lattice_a_angstrom <= 0.0 or self.lattice_c_angstrom <= 0.0:
            raise ValueError("lattice constants must be positive")
        site_gap = float(self.site_energies_eV["O"] - self.site_energies_eV["T"])
        barrier_gap = float(
            self.migration_barriers_eV["TO"] - self.migration_barriers_eV["OT"]
        )
        if not np.isclose(site_gap, barrier_gap, atol=1.0e-8):
            raise ValueError(
                "TO/OT barrier difference must equal the O/T site-energy difference "
                f"for a shared transition state; got {barrier_gap} versus {site_gap} eV"
            )
        return self

    def rate(self, hop_kind, temperature_K):
        hop_kind = str(hop_kind).upper()
        if hop_kind not in REQUIRED_HOPS:
            raise KeyError(f"unknown H hop kind {hop_kind!r}")
        temperature_K = float(temperature_K)
        if temperature_K <= 0.0:
            raise ValueError("temperature must be positive")
        k_B_eV_per_K = 8.617333262145e-5
        return float(
            self.attempt_frequencies_Hz[hop_kind]
            * np.exp(
                -self.migration_barriers_eV[hop_kind] / (k_B_eV_per_K * temperature_K)
            )
        )

    def metadata(self):
        return {
            "name": self.name,
            "material_system": "hcp_zr_h",
            "host_phase": "alpha_zr",
            "source": self.source,
            "publication_ready": bool(self.publication_ready),
            "site_energies_eV": dict(self.site_energies_eV),
            "migration_barriers_eV": dict(self.migration_barriers_eV),
            "attempt_frequencies_Hz": dict(self.attempt_frequencies_Hz),
            "lattice_constants_angstrom": {
                "a": self.lattice_a_angstrom,
                "c": self.lattice_c_angstrom,
            },
        }


def _default_parameter_path():
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "hcp"
        / "zr_h_dilute_zhang2017.json"
    )


def load_zrh_parameters(path=None):
    parameter_path = (
        Path(path).expanduser().resolve() if path else _default_parameter_path()
    )
    payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    if payload.get("material_system") != "hcp_zr_h":
        raise ValueError("parameter file material_system must be 'hcp_zr_h'")
    if payload.get("host_phase") != "alpha_zr":
        raise ValueError("parameter file host_phase must be 'alpha_zr'")
    if payload.get("energy_unit") != "eV":
        raise ValueError("parameter file energy_unit must be 'eV'")
    hops = payload.get("hops", {})
    lattice = payload.get("lattice_constants_angstrom", {})
    params = ZrHKMCParameters(
        name=str(payload.get("name", parameter_path.stem)),
        site_energies_eV={
            key: float(value)
            for key, value in payload.get("site_energies_eV", {}).items()
        },
        migration_barriers_eV={
            key: float(hops[key]["migration_barrier_eV"]) for key in REQUIRED_HOPS
        },
        attempt_frequencies_Hz={
            key: float(hops[key]["attempt_frequency_Hz"]) for key in REQUIRED_HOPS
        },
        lattice_a_angstrom=float(lattice["a"]),
        lattice_c_angstrom=float(lattice["c"]),
        source=str(payload.get("source", parameter_path)),
        publication_ready=bool(payload.get("publication_ready", False)),
    )
    return params.validate()
