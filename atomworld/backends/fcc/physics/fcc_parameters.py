"""Parameter loading and validation for vacancy-mediated FCC Ni-Al-Cr KMC."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SPECIES = ("Ni", "Al", "Cr", "Vacancy")
MOBILE_SPECIES = SPECIES[:3]


@dataclass(frozen=True)
class FCCKMCParameters:
    name: str
    pair_energies: np.ndarray
    migration_barriers: np.ndarray
    saddle_point_energies: np.ndarray
    attempt_frequencies: np.ndarray
    rate_model: str
    energy_unit: str
    source: str
    publication_ready: bool

    def validate(self):
        pair = np.asarray(self.pair_energies, dtype=np.float64)
        barriers = np.asarray(self.migration_barriers, dtype=np.float64)
        saddle = np.asarray(self.saddle_point_energies, dtype=np.float64)
        attempts = np.asarray(self.attempt_frequencies, dtype=np.float64)
        if pair.shape != (4, 4, 4):
            raise ValueError(f"pair_energies must have shape (4, 4, 4), got {pair.shape}")
        if barriers.shape != (3,):
            raise ValueError(f"migration_barriers must have shape (3,), got {barriers.shape}")
        if saddle.shape != (3,):
            raise ValueError(f"saddle_point_energies must have shape (3,), got {saddle.shape}")
        if attempts.shape != (3,):
            raise ValueError(f"attempt_frequencies must have shape (3,), got {attempts.shape}")
        if not np.isfinite(pair).all() or not np.isfinite(barriers).all() or not np.isfinite(saddle).all():
            raise ValueError("FCC KMC parameters must be finite")
        if not np.allclose(pair, pair.transpose(0, 2, 1), atol=1.0e-12):
            raise ValueError("Each neighbor-shell pair-energy matrix must be symmetric")
        if np.any(attempts <= 0.0) or not np.isfinite(attempts).all():
            raise ValueError("Attempt frequencies must be finite and strictly positive")
        if self.rate_model == "symmetric_kra":
            if np.any(barriers <= 0.0):
                raise ValueError("Migration barriers must be strictly positive for symmetric_kra")
        elif self.rate_model != "broken_bond_saddle":
            raise ValueError(f"Unsupported FCC rate_model={self.rate_model!r}")
        if self.energy_unit != "eV":
            raise ValueError(f"Only eV parameter files are supported, got {self.energy_unit!r}")
        return self

    def metadata(self):
        return {
            "name": self.name,
            "energy_unit": self.energy_unit,
            "source": self.source,
            "publication_ready": bool(self.publication_ready),
            "rate_model": self.rate_model,
            "attempt_frequencies_s^-1": self.attempt_frequencies.tolist(),
            "saddle_point_energies_eV": self.saddle_point_energies.tolist(),
            "species_order": list(SPECIES),
            "mobile_species_order": list(MOBILE_SPECIES),
        }


def _debug_parameters():
    first_shell = np.asarray(
        [
            [-0.095, -0.105, -0.090, -0.010],
            [-0.105, -0.065, -0.080, -0.006],
            [-0.090, -0.080, -0.075, -0.008],
            [-0.010, -0.006, -0.008, 0.020],
        ],
        dtype=np.float64,
    )
    pair = np.stack([scale * first_shell for scale in (1.0, 0.25, 0.10, 0.05)], axis=0)
    return FCCKMCParameters(
        name="debug",
        pair_energies=pair,
        migration_barriers=np.asarray([1.05, 0.85, 1.15], dtype=np.float64),
        saddle_point_energies=np.zeros(3, dtype=np.float64),
        attempt_frequencies=np.full(3, 1.0e13, dtype=np.float64),
        rate_model="symmetric_kra",
        energy_unit="eV",
        source="Internal numerical-debug values; not fitted to Ni-Al-Cr data.",
        publication_ready=False,
    ).validate()


def _from_json(path):
    parameter_path = Path(path).expanduser().resolve()
    payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    if payload.get("material_system") != "fcc_ni_al_cr":
        raise ValueError("Parameter file material_system must be 'fcc_ni_al_cr'")
    if payload.get("species_order") != list(SPECIES):
        raise ValueError(f"Parameter file species_order must be {list(SPECIES)}")
    barriers = payload.get("migration_barriers", {})
    saddle = payload.get("saddle_point_energies", {})
    attempts = payload.get("attempt_frequencies", {})
    rate_model = str(payload.get("rate_model", "symmetric_kra"))
    params = FCCKMCParameters(
        name=str(payload.get("name", parameter_path.stem)),
        pair_energies=np.asarray(payload["pair_energies"], dtype=np.float64),
        migration_barriers=np.asarray(
            [barriers.get(name, 0.0) for name in MOBILE_SPECIES], dtype=np.float64
        ),
        saddle_point_energies=np.asarray(
            [saddle.get(name, 0.0) for name in MOBILE_SPECIES], dtype=np.float64
        ),
        attempt_frequencies=np.asarray(
            [attempts.get(name, 1.0e13) for name in MOBILE_SPECIES], dtype=np.float64
        ),
        rate_model=rate_model,
        energy_unit=str(payload.get("energy_unit", "")),
        source=str(payload.get("source", parameter_path)),
        publication_ready=bool(payload.get("publication_ready", False)),
    )
    return params.validate()


def load_fcc_parameters(args):
    parameter_file = str(getattr(args, "fcc_parameter_file", "") or "").strip()
    parameter_set = str(getattr(args, "fcc_parameter_set", "debug") or "debug").strip().lower()
    if parameter_file:
        params = _from_json(parameter_file)
    elif parameter_set == "debug":
        params = _debug_parameters()
    elif parameter_set == "mao2012_lkmc1":
        params = _from_json(
            Path(__file__).resolve().parents[1] / "configs" / "fcc" / "ni_al_cr_mao2012_lkmc1.json"
        )
    else:
        raise ValueError(
            f"Unknown fcc_parameter_set={parameter_set!r}; expected 'debug' or 'mao2012_lkmc1', "
            "or provide --fcc_parameter_file"
        )

    overrides = np.asarray(params.migration_barriers, dtype=np.float64).copy()
    for index, attr in enumerate(("ni_migration_barrier", "al_migration_barrier", "cr_migration_barrier")):
        value = getattr(args, attr, None)
        if value is not None:
            if params.rate_model != "symmetric_kra":
                raise ValueError(
                    f"--{attr} is only valid for rate_model='symmetric_kra'; "
                    f"{params.name!r} uses {params.rate_model!r}"
                )
            overrides[index] = float(value)
    overridden = FCCKMCParameters(
        name=params.name,
        pair_energies=np.asarray(params.pair_energies, dtype=np.float64).copy(),
        migration_barriers=overrides,
        saddle_point_energies=np.asarray(params.saddle_point_energies, dtype=np.float64).copy(),
        attempt_frequencies=np.asarray(params.attempt_frequencies, dtype=np.float64).copy(),
        rate_model=params.rate_model,
        energy_unit=params.energy_unit,
        source=params.source,
        publication_ready=params.publication_ready,
    )
    return overridden.validate()
