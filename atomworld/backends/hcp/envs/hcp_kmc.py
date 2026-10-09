"""Passive residence-time lattice KMC for interstitial H in alpha-Zr."""

from __future__ import annotations

from collections import Counter

import numpy as np

from atomworld.backends.hcp.lattices.hcp_interstitial import (
    HCPInterstitialLattice,
    HOP_KINDS,
    OCTAHEDRAL,
    TETRAHEDRAL,
)
from atomworld.backends.hcp.physics.events import EventCatalog, TransitionResult
from atomworld.backends.hcp.physics.local_energy import LocalEnergyModel
from atomworld.backends.hcp.physics.zrh_parameters import load_zrh_parameters


EVENT_FEATURE_DIM = 18
K_B_EV_PER_K = 8.617333262145e-5


class HCPHydrogenKMC:
    """Physics-only HCP KMC environment.

    This class always samples an event in proportion to its physical rate.
    Consequently, and only consequently, its ordinary residence-time clock is
    valid.  A learned policy will use the same catalogue and transition
    operator but a separate clock implementation.
    """

    def __init__(
        self,
        *,
        size=(6, 6, 6),
        n_hydrogen=1,
        temperature_K=600.0,
        seed=1,
        parameter_file=None,
        initial_site_type="T",
        energy_model=None,
    ):
        self.parameters = load_zrh_parameters(parameter_file)
        self.temperature_K = float(temperature_K)
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        self.seed = int(seed)
        self.energy_model = LocalEnergyModel() if energy_model is None else energy_model
        if not isinstance(self.energy_model, LocalEnergyModel):
            raise TypeError("energy_model must be a LocalEnergyModel")
        self.lattice = HCPInterstitialLattice(
            size,
            lattice_a_angstrom=self.parameters.lattice_a_angstrom,
            lattice_c_angstrom=self.parameters.lattice_c_angstrom,
            n_hydrogen=n_hydrogen,
            seed=self.seed,
            initial_site_type=initial_site_type,
        )
        self.rng = np.random.default_rng(self.seed + 104729)
        self.time_s = 0.0
        self.step_count = 0
        self.hop_counts = Counter({kind: 0 for kind in HOP_KINDS})
        self.site_residence_time_s = Counter({"T": 0.0, "O": 0.0})
        self.previous_transition = None
        self.immediate_reverse_count = 0
        self.energy_eV = self.calculate_energy_eV()

    @property
    def num_agents(self):
        return self.lattice.n_hydrogen

    @property
    def events_per_agent(self):
        return self.lattice.events_per_hydrogen

    def calculate_energy_eV(self):
        names = np.where(
            self.lattice.site_types[self.lattice.hydrogen_sites] == TETRAHEDRAL,
            "T",
            "O",
        )
        intrinsic = float(
            sum(self.parameters.site_energies_eV[str(name)] for name in names)
        )
        return intrinsic + self.energy_model.total_extra_energy_eV(self.lattice)

    def event_catalog(self):
        shape = (self.num_agents, self.events_per_agent)
        rates = np.zeros(shape, dtype=np.float64)
        features = np.zeros((*shape, EVENT_FEATURE_DIM), dtype=np.float32)
        for hydrogen_index, source in enumerate(self.lattice.hydrogen_sites):
            source = int(source)
            source_is_t = int(self.lattice.site_types[source]) == TETRAHEDRAL
            for local_event in range(self.events_per_agent):
                target = int(self.lattice.event_targets[source, local_event])
                kind_id = int(self.lattice.event_kinds[source, local_event])
                if target < 0 or kind_id < 0:
                    continue
                kind = HOP_KINDS[kind_id]
                target_is_t = int(self.lattice.site_types[target]) == TETRAHEDRAL
                legal = self.lattice.occupancy[target] < 0
                delta_energy = float(
                    self.parameters.site_energies_eV["T" if target_is_t else "O"]
                    - self.parameters.site_energies_eV["T" if source_is_t else "O"]
                )
                barrier = float(self.parameters.migration_barriers_eV[kind])
                physical_rate = self.parameters.rate(kind, self.temperature_K)
                displacement = self.lattice.event_displacements_angstrom[
                    source, local_event
                ]

                trap_delta = 0.0
                pair_delta = 0.0
                coordination_target = 0
                if legal and self.energy_model.enabled:
                    extra = self.energy_model.event_extra_energy_change_eV(
                        self.lattice, hydrogen_index, target
                    )
                    trap_delta = extra["trap_delta_eV"]
                    pair_delta = extra["pair_delta_eV"]
                    coordination_target = extra["coordination_target"]
                extra_delta = trap_delta + pair_delta
                effective_barrier = barrier + max(0.0, extra_delta)
                corrected_rate = physical_rate * np.exp(
                    -max(0.0, extra_delta) / (K_B_EV_PER_K * self.temperature_K)
                )
                rate = corrected_rate if legal else 0.0

                features[hydrogen_index, local_event, 0] = float(legal)
                features[hydrogen_index, local_event, 1 + kind_id] = 1.0
                features[hydrogen_index, local_event, 5:8] = displacement
                features[hydrogen_index, local_event, 8] = float(
                    np.linalg.norm(displacement)
                )
                features[hydrogen_index, local_event, 9] = float(source_is_t)
                features[hydrogen_index, local_event, 10] = float(target_is_t)
                features[hydrogen_index, local_event, 11] = delta_energy + extra_delta
                features[hydrogen_index, local_event, 12] = effective_barrier
                features[hydrogen_index, local_event, 13] = float(
                    np.log(max(corrected_rate, 1.0e-300))
                )
                features[hydrogen_index, local_event, 14] = float(coordination_target)
                features[hydrogen_index, local_event, 15] = trap_delta
                features[hydrogen_index, local_event, 16] = pair_delta
                features[hydrogen_index, local_event, 17] = delta_energy
                rates[hydrogen_index, local_event] = rate
        return EventCatalog(rates, features)

    def apply_event(
        self,
        event_id,
        *,
        catalog=None,
        delta_time_s=0.0,
        clock_owner="unassigned",
    ):
        clock_owner = str(clock_owner)
        if clock_owner not in {"unassigned", "passive_kmc", "policy_poisson"}:
            raise ValueError(f"unsupported clock_owner={clock_owner!r}")
        if clock_owner == "unassigned" and float(delta_time_s) != 0.0:
            raise ValueError("an unassigned transition cannot advance physical time")
        if float(delta_time_s) < 0.0 or not np.isfinite(float(delta_time_s)):
            raise ValueError("delta_time_s must be finite and non-negative")
        catalog = self.event_catalog() if catalog is None else catalog
        hydrogen_index, local_event = catalog.assert_legal(int(event_id))
        source = int(self.lattice.hydrogen_sites[hydrogen_index])
        target = int(self.lattice.event_targets[source, local_event])
        kind = HOP_KINDS[int(self.lattice.event_kinds[source, local_event])]
        energy_before = float(self.energy_eV)
        source_name = self.lattice.site_type_name(source)
        target_name = self.lattice.site_type_name(target)
        intrinsic_delta = float(
            self.parameters.site_energies_eV[target_name]
            - self.parameters.site_energies_eV[source_name]
        )
        extra = {
            "trap_delta_eV": 0.0,
            "pair_delta_eV": 0.0,
            "coordination_target": 0,
        }
        if self.energy_model.enabled:
            extra = self.energy_model.event_extra_energy_change_eV(
                self.lattice, hydrogen_index, target
            )
        exact_delta = intrinsic_delta + extra["trap_delta_eV"] + extra["pair_delta_eV"]
        immediate_reverse = bool(
            self.previous_transition is not None
            and self.previous_transition[0] == hydrogen_index
            and self.previous_transition[1] == target
            and self.previous_transition[2] == source
        )
        source, target, displacement = self.lattice.move_hydrogen(
            hydrogen_index, local_event
        )
        self.energy_eV += float(exact_delta)
        self.time_s += float(delta_time_s)
        self.step_count += 1
        self.hop_counts[kind] += 1
        self.immediate_reverse_count += int(immediate_reverse)
        self.previous_transition = (int(hydrogen_index), int(source), int(target))
        self.lattice.assert_consistent()
        return TransitionResult(
            event_id=int(event_id),
            hydrogen_index=int(hydrogen_index),
            local_event_index=int(local_event),
            source_site=int(source),
            target_site=int(target),
            hop_kind=kind,
            delta_time_s=float(delta_time_s),
            energy_before_eV=energy_before,
            energy_after_eV=float(self.energy_eV),
            displacement_angstrom=displacement,
            info={
                "temperature_K": self.temperature_K,
                "clock_owner": clock_owner,
                "passive_kmc": clock_owner == "passive_kmc",
                "immediate_reverse": immediate_reverse,
                "intrinsic_delta_eV": intrinsic_delta,
                **extra,
            },
        )

    def step(self):
        catalog = self.event_catalog()
        total_rate = catalog.total_rate
        if total_rate <= 0.0:
            raise RuntimeError("HCP KMC state exposes no legal hydrogen event")
        threshold = float(self.rng.random()) * total_rate
        cumulative = np.cumsum(catalog.rates.reshape(-1), dtype=np.float64)
        event_id = min(
            int(np.searchsorted(cumulative, threshold, side="right")),
            cumulative.size - 1,
        )
        uniform = max(float(self.rng.random()), np.finfo(np.float64).tiny)
        delta_time = -float(np.log(uniform)) / total_rate
        occupied_types = self.lattice.site_types[self.lattice.hydrogen_sites]
        self.site_residence_time_s["T"] += delta_time * float(
            np.count_nonzero(occupied_types == TETRAHEDRAL)
        )
        self.site_residence_time_s["O"] += delta_time * float(
            np.count_nonzero(occupied_types == OCTAHEDRAL)
        )
        return self.apply_event(
            event_id,
            catalog=catalog,
            delta_time_s=delta_time,
            clock_owner="passive_kmc",
        )

    def run(self, steps):
        last = None
        for _ in range(int(steps)):
            last = self.step()
        return last

    def diffusion_coefficients_m2_per_s(self):
        if self.time_s <= 0.0:
            return {"D_a": 0.0, "D_c": 0.0, "D_3d": 0.0, "D_c_over_D_a": np.nan}
        displacement_m = self.lattice.unwrapped_displacements_angstrom * 1.0e-10
        basal_msd = float(np.mean(np.sum(displacement_m[:, :2] ** 2, axis=1)))
        c_msd = float(np.mean(displacement_m[:, 2] ** 2))
        total_msd = basal_msd + c_msd
        d_a = basal_msd / (4.0 * self.time_s)
        d_c = c_msd / (2.0 * self.time_s)
        d_3d = total_msd / (6.0 * self.time_s)
        return {
            "D_a": d_a,
            "D_c": d_c,
            "D_3d": d_3d,
            "D_c_over_D_a": d_c / d_a if d_a > 0.0 else np.nan,
        }

    def analytical_dilute_diffusivity_m2_per_s(self):
        """Ishioka-Koiwa expressions used by Zhang et al. for H in HCP Zr."""
        w_tt = self.parameters.rate("TT", self.temperature_K)
        w_to = self.parameters.rate("TO", self.temperature_K)
        w_ot = self.parameters.rate("OT", self.temperature_K)
        w_oo = self.parameters.rate("OO", self.temperature_K)
        a2 = (self.parameters.lattice_a_angstrom * 1.0e-10) ** 2
        c2 = (self.parameters.lattice_c_angstrom * 1.0e-10) ** 2
        d_a = (w_to * w_ot / (w_to + 2.0 * w_ot)) * a2
        numerator = w_to * (3.0 * w_oo * w_to + 2.0 * w_oo * w_tt + 3.0 * w_tt * w_ot)
        denominator = 4.0 * (2.0 * w_tt + 3.0 * w_to) * (w_to + 2.0 * w_ot)
        d_c = (numerator / denominator) * c2
        return {
            "D_a": float(d_a),
            "D_c": float(d_c),
            "D_3d": float((2.0 * d_a + d_c) / 3.0),
            "D_c_over_D_a": float(d_c / d_a),
        }

    def site_residence_fractions(self):
        total = float(sum(self.site_residence_time_s.values()))
        if total <= 0.0:
            return {"T": np.nan, "O": np.nan}
        return {
            "T": float(self.site_residence_time_s["T"] / total),
            "O": float(self.site_residence_time_s["O"] / total),
        }

    def analytical_site_fractions(self):
        # Stationary T<->O flux balance: P_T * 3*w_TO = P_O * 6*w_OT.
        ratio_o_over_t = self.parameters.rate("TO", self.temperature_K) / (
            2.0 * self.parameters.rate("OT", self.temperature_K)
        )
        p_t = 1.0 / (1.0 + ratio_o_over_t)
        return {"T": float(p_t), "O": float(1.0 - p_t)}

    def analytical_hop_fractions(self):
        site_fractions = self.analytical_site_fractions()
        propensities = {
            "TT": site_fractions["T"] * self.parameters.rate("TT", self.temperature_K),
            "TO": site_fractions["T"]
            * 3.0
            * self.parameters.rate("TO", self.temperature_K),
            "OT": site_fractions["O"]
            * 6.0
            * self.parameters.rate("OT", self.temperature_K),
            "OO": site_fractions["O"]
            * 2.0
            * self.parameters.rate("OO", self.temperature_K),
        }
        total = float(sum(propensities.values()))
        return {kind: float(propensities[kind] / total) for kind in HOP_KINDS}

    def analytical_dilute_keyframe_first_passage(self):
        """Conditional means for the default passive keyframe definition.

        Starting at T, TT self-flicker competes with three TO escape events.
        Starting at O, all two OO and six OT events are keyframes.
        """
        w_tt = self.parameters.rate("TT", self.temperature_K)
        w_to = self.parameters.rate("TO", self.temperature_K)
        w_ot = self.parameters.rate("OT", self.temperature_K)
        w_oo = self.parameters.rate("OO", self.temperature_K)
        t_total = w_tt + 3.0 * w_to
        o_total = 2.0 * w_oo + 6.0 * w_ot
        return {
            "T": {
                "mean_microsteps": float(t_total / (3.0 * w_to)),
                "mean_elapsed_physical_time_s": float(1.0 / (3.0 * w_to)),
            },
            "O": {
                "mean_microsteps": 1.0,
                "mean_elapsed_physical_time_s": float(1.0 / o_total),
            },
        }

    def kinetic_diagnostics(self):
        total_hops = int(sum(self.hop_counts.values()))
        return {
            "hop_fractions": {
                kind: float(self.hop_counts[kind] / max(total_hops, 1))
                for kind in HOP_KINDS
            },
            "immediate_reverse_fraction": float(
                self.immediate_reverse_count / max(total_hops - 1, 1)
            ),
            "immediate_reverse_count": int(self.immediate_reverse_count),
            "total_hops": total_hops,
        }

    def state_dict(self):
        return {
            "schema_version": 1,
            "material_system": "hcp_zr_h",
            "host_phase": "alpha_zr",
            "size": tuple(self.lattice.size),
            "temperature_K": float(self.temperature_K),
            "hydrogen_sites": self.lattice.hydrogen_sites.copy(),
            "occupancy": self.lattice.occupancy.copy(),
            "unwrapped_displacements_angstrom": self.lattice.unwrapped_displacements_angstrom.copy(),
            "time_s": float(self.time_s),
            "step_count": int(self.step_count),
            "energy_eV": float(self.energy_eV),
            "hop_counts": dict(self.hop_counts),
            "site_residence_time_s": dict(self.site_residence_time_s),
            "previous_transition": self.previous_transition,
            "immediate_reverse_count": int(self.immediate_reverse_count),
            "rng_state": self.rng.bit_generator.state,
            "parameter_metadata": self.parameters.metadata(),
            "energy_model_metadata": self.energy_model.metadata(),
        }

    def load_state_dict(self, state):
        if (
            state.get("material_system") != "hcp_zr_h"
            or state.get("host_phase") != "alpha_zr"
        ):
            raise ValueError("checkpoint is not an alpha-Zr/H state")
        if tuple(state.get("size", ())) != tuple(self.lattice.size):
            raise ValueError(
                "checkpoint supercell size does not match this environment"
            )
        if state.get("energy_model_metadata") != self.energy_model.metadata():
            raise ValueError(
                "checkpoint local-energy model does not match this environment"
            )
        hydrogen_sites = np.asarray(state["hydrogen_sites"], dtype=np.int64)
        occupancy = np.asarray(state["occupancy"], dtype=np.int64)
        displacements = np.asarray(
            state["unwrapped_displacements_angstrom"], dtype=np.float64
        )
        if hydrogen_sites.shape != self.lattice.hydrogen_sites.shape:
            raise ValueError(
                "checkpoint hydrogen count does not match this environment"
            )
        if occupancy.shape != self.lattice.occupancy.shape:
            raise ValueError("checkpoint occupancy shape mismatch")
        if displacements.shape != self.lattice.unwrapped_displacements_angstrom.shape:
            raise ValueError("checkpoint displacement shape mismatch")
        self.lattice.hydrogen_sites[...] = hydrogen_sites
        self.lattice.occupancy[...] = occupancy
        self.lattice.unwrapped_displacements_angstrom[...] = displacements
        self.temperature_K = float(state["temperature_K"])
        self.time_s = float(state["time_s"])
        self.step_count = int(state["step_count"])
        self.energy_eV = float(state["energy_eV"])
        self.hop_counts = Counter(
            {kind: int(state["hop_counts"].get(kind, 0)) for kind in HOP_KINDS}
        )
        residence = state.get("site_residence_time_s", {})
        self.site_residence_time_s = Counter(
            {"T": float(residence.get("T", 0.0)), "O": float(residence.get("O", 0.0))}
        )
        previous = state.get("previous_transition")
        self.previous_transition = (
            tuple(map(int, previous)) if previous is not None else None
        )
        self.immediate_reverse_count = int(state.get("immediate_reverse_count", 0))
        self.rng.bit_generator.state = state["rng_state"]
        self.lattice.assert_consistent()
