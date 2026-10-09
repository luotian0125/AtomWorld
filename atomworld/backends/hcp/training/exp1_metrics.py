"""Experiment-1 metrics for fixed-event HCP main-table comparisons.

The metrics in this module deliberately own no physical clock.  They answer
whether a matched number of simulator-admissible interstitial-H events creates
more persistent transport and dissolved-H precursor evolution.  Dynamics,
structure, and clock fidelity are reserved for a separately calibrated stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from atomworld.backends.hcp.learning.hcp_binding import HCPAtomWorldBinding
from atomworld.backends.hcp.training.reward import HCPRewardModel


BASELINE_METHODS = (
    "classical_hcp_lkmc",
    "amc_superbasin_reference",
    "rate_greedy_tabu",
    "energy_greedy_tabu",
)


@dataclass(frozen=True)
class PersistentPrecursorDefinition:
    """Auditable dissolved-H precursor-surrogate definition.

    This does not classify a hydride phase.  A hit must satisfy all component
    thresholds for ``persistence_samples`` consecutive sampled observations.
    """

    min_attractive_pairs_per_h: float = 0.40
    min_largest_cluster_fraction: float = 0.14
    min_nonisolated_fraction: float = 0.70
    persistence_samples: int = 4

    def __post_init__(self):
        for name in (
            "min_attractive_pairs_per_h",
            "min_largest_cluster_fraction",
            "min_nonisolated_fraction",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.persistence_samples <= 0:
            raise ValueError("persistence_samples must be positive")

    def reached(self, snapshot):
        return bool(
            float(snapshot["attractive_pairs_per_h"])
            >= self.min_attractive_pairs_per_h
            and float(snapshot["largest_cluster_fraction"])
            >= self.min_largest_cluster_fraction
            and float(snapshot["nonisolated_fraction"])
            >= self.min_nonisolated_fraction
        )

    def metadata(self):
        payload = asdict(self)
        payload["claim_boundary"] = (
            "persistent dissolved-H precursor surrogate; not a hydride phase label"
        )
        return payload


def precursor_coordinate(snapshot):
    """Non-negative structural coordinate used for the precursor AUC.

    Energy and policy reward are intentionally excluded.  The coordinate is
    still a dissolved-H surrogate and must be accompanied by the component
    values and persistence criterion in publication-facing output.
    """

    return float(
        0.25 * float(snapshot["attractive_pairs_per_h"])
        + float(snapshot["largest_cluster_fraction"])
        + 0.5 * float(snapshot["nonisolated_fraction"])
    )


def persistent_transport_coordinate(kmc):
    """Mean unwrapped squared displacement in A^2 per hydrogen.

    Unlike accumulated path length, a direct T->T->T recross returns to its
    prior displacement and therefore does not create artificial progress.
    """

    displacement = np.asarray(
        kmc.lattice.unwrapped_displacements_angstrom, dtype=np.float64
    )
    return float(np.mean(np.sum(displacement * displacement, axis=1)))


def _reverse_filtered_ids(env, catalog):
    legal_ids = np.flatnonzero(catalog.legal_mask.reshape(-1))
    previous = env.kmc.previous_transition
    if previous is None:
        return legal_ids
    previous_agent, previous_source, previous_target = map(int, previous)
    kept = []
    for event_id in legal_ids:
        agent, local_event = catalog.decode(int(event_id))
        source = int(env.kmc.lattice.hydrogen_sites[agent])
        target = int(env.kmc.lattice.event_targets[source, local_event])
        reverse = bool(
            agent == previous_agent
            and source == previous_target
            and target == previous_source
        )
        if not reverse:
            kept.append(int(event_id))
    return np.asarray(kept if kept else legal_ids, dtype=np.int64)


def select_tabu_baseline_event(env, catalog, method, rng):
    """Select a strong instantaneous baseline with one-step reverse tabu."""

    legal_ids = _reverse_filtered_ids(env, catalog)
    if legal_ids.size == 0:
        raise RuntimeError("HCP state contains no legal event")
    rates = catalog.rates.reshape(-1)
    if method == "rate_greedy_tabu":
        return int(legal_ids[np.argmax(rates[legal_ids])])
    if method == "energy_greedy_tabu":
        delta_energy = catalog.features[..., 11].reshape(-1)
        best_delta = np.min(delta_energy[legal_ids])
        tied = legal_ids[np.isclose(delta_energy[legal_ids], best_delta)]
        return int(tied[np.argmax(rates[tied])])
    raise ValueError(f"unsupported tabu baseline {method!r}")


def _trapezoid_mean(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x[-1] <= x[0]:
        return float(y[-1]) if y.size else 0.0
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / (x[-1] - x[0]))


def evaluate_exp1_episode(
    *,
    environment_factory,
    reward_model,
    seed,
    steps,
    sample_interval,
    precursor_definition,
    method,
    model=None,
    model_config=None,
    device="cpu",
    policy_temperature=1.0,
):
    """Run one matched fixed-event trajectory and retain auditable samples."""

    if steps <= 0 or sample_interval <= 0:
        raise ValueError("steps and sample_interval must be positive")
    env = environment_factory(int(seed))
    rng = np.random.default_rng(int(seed) + 49979687)
    binding = None
    previous_long_state = None
    update_mask = np.ones(env.num_agents, dtype=bool)
    if model is not None:
        binding = HCPAtomWorldBinding(
            env, config=model_config, device=device, model=model
        )

    event_axis = [0]
    transport_values = [persistent_transport_coordinate(env.kmc)]
    initial_snapshot = reward_model.snapshot(env.kmc)
    precursor_values = [precursor_coordinate(initial_snapshot)]
    component_samples = [
        {
            "event": 0,
            "transport_msd_A2": transport_values[-1],
            "precursor_coordinate": precursor_values[-1],
            **{key: float(value) for key, value in initial_snapshot.items()},
        }
    ]
    consecutive_precursor_samples = int(
        precursor_definition.reached(initial_snapshot)
    )
    first_persistent_hit_event = None
    tt_events = 0
    tt_recrosses = 0
    all_immediate_recrosses = 0
    represented_tt_events = 0
    represented_tt_recrosses = 0
    hop_counts = {kind: 0 for kind in ("TT", "TO", "OT", "OO")}

    for event_number in range(1, int(steps) + 1):
        if method in {"classical_hcp_lkmc", "amc_superbasin_reference"}:
            physical = env.kmc.step()
            kind = str(physical.hop_kind)
            reverse = bool(physical.info.get("immediate_reverse", False))
            # The reference projection explicitly executes hidden microsteps,
            # but an AMC/superbasin trajectory exposes only basin exits.  It is
            # therefore exact for fixed micro-event progress, while deliberately
            # not claiming an optimized AMC implementation or wall-clock gain.
            represented = method == "classical_hcp_lkmc" or kind != "TT"
        else:
            catalog = env.enumerate_events()
            if binding is not None:
                event_id, next_long_state = binding.select_event(
                    previous_long_state=previous_long_state,
                    update_mask=update_mask,
                    deterministic=True,
                    policy_temperature=policy_temperature,
                )
            else:
                event_id = select_tabu_baseline_event(env, catalog, method, rng)
                next_long_state = None
            policy_transition = env.execute_event(event_id)
            physical = policy_transition.physical_transition
            kind = str(physical.hop_kind)
            reverse = bool(physical.info.get("immediate_reverse", False))
            represented = True
            if binding is not None:
                previous_long_state = next_long_state.detach()
                update_mask = policy_transition.long_memory_update_mask

        hop_counts[kind] += 1
        all_immediate_recrosses += int(reverse)
        if kind == "TT":
            tt_events += 1
            tt_recrosses += int(reverse)
        if represented and kind == "TT":
            represented_tt_events += 1
            represented_tt_recrosses += int(reverse)

        if event_number % sample_interval == 0 or event_number == int(steps):
            snapshot = reward_model.snapshot(env.kmc)
            transport = persistent_transport_coordinate(env.kmc)
            precursor = precursor_coordinate(snapshot)
            event_axis.append(event_number)
            transport_values.append(transport)
            precursor_values.append(precursor)
            component_samples.append(
                {
                    "event": int(event_number),
                    "transport_msd_A2": transport,
                    "precursor_coordinate": precursor,
                    **{key: float(value) for key, value in snapshot.items()},
                }
            )
            if precursor_definition.reached(snapshot):
                consecutive_precursor_samples += 1
                if (
                    first_persistent_hit_event is None
                    and consecutive_precursor_samples
                    >= precursor_definition.persistence_samples
                ):
                    first_persistent_hit_event = int(
                        event_number
                        - (precursor_definition.persistence_samples - 1)
                        * sample_interval
                    )
            else:
                consecutive_precursor_samples = 0

    final_snapshot = reward_model.snapshot(env.kmc)
    table_recrosses = (
        represented_tt_recrosses
        if method == "amc_superbasin_reference"
        else tt_recrosses
    )
    table_tt_events = (
        represented_tt_events
        if method == "amc_superbasin_reference"
        else tt_events
    )
    return {
        "method": str(method),
        "seed": int(seed),
        "events": int(steps),
        "events_per_h": float(steps / max(env.num_agents, 1)),
        "transport_auc_A2": _trapezoid_mean(event_axis, transport_values),
        "precursor_auc": _trapezoid_mean(event_axis, precursor_values),
        "persistent_precursor_hit": first_persistent_hit_event is not None,
        "persistent_precursor_hit_event": first_persistent_hit_event,
        "persistent_precursor_hit_q": (
            None
            if first_persistent_hit_event is None
            else float(first_persistent_hit_event / max(env.num_agents, 1))
        ),
        "tt_events": int(table_tt_events),
        "tt_recrosses": int(table_recrosses),
        "tt_recross_fraction": float(
            table_recrosses / max(table_tt_events, 1)
        ),
        "underlying_tt_events": int(tt_events),
        "underlying_tt_recrosses": int(tt_recrosses),
        "underlying_tt_recross_fraction": float(tt_recrosses / max(tt_events, 1)),
        "all_immediate_recross_fraction": float(
            all_immediate_recrosses / max(int(steps), 1)
        ),
        "hop_counts": hop_counts,
        "final_precursor_coordinate": precursor_coordinate(final_snapshot),
        "final_attractive_pairs_per_h": float(
            final_snapshot["attractive_pairs_per_h"]
        ),
        "final_largest_cluster_fraction": float(
            final_snapshot["largest_cluster_fraction"]
        ),
        "final_nonisolated_fraction": float(
            final_snapshot["nonisolated_fraction"]
        ),
        "samples": component_samples,
        "clock_calibrated": False,
    }


def kaplan_meier_median(hit_events, *, censor_event):
    """Return a simple Kaplan-Meier median event count, or ``None``."""

    observations = sorted(
        (
            int(value) if value is not None else int(censor_event),
            value is not None,
        )
        for value in hit_events
    )
    at_risk = len(observations)
    survival = 1.0
    for event in sorted({item[0] for item in observations}):
        deaths = sum(time == event and hit for time, hit in observations)
        censored = sum(time == event and not hit for time, hit in observations)
        if at_risk > 0 and deaths:
            survival *= 1.0 - deaths / at_risk
            if survival <= 0.5:
                return int(event)
        at_risk -= deaths + censored
    return None


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else math.nan
    if values.size <= 1:
        return mean, mean, mean
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def summarize_exp1_episodes(episodes, *, classical_method, steps):
    """Create main-table rows while leaving uncalibrated fidelity fields null."""

    grouped = {}
    for episode in episodes:
        grouped.setdefault(episode["method"], []).append(episode)
    if classical_method not in grouped:
        raise ValueError("classical reference episodes are required")
    reference_transport = np.mean(
        [row["transport_auc_A2"] for row in grouped[classical_method]]
    )
    reference_precursor = np.mean(
        [row["precursor_auc"] for row in grouped[classical_method]]
    )
    reference_q50 = kaplan_meier_median(
        [row["persistent_precursor_hit_event"] for row in grouped[classical_method]],
        censor_event=steps,
    )
    rows = []
    for method, values in grouped.items():
        transport = _mean_ci([row["transport_auc_A2"] for row in values])
        precursor = _mean_ci([row["precursor_auc"] for row in values])
        hit_rate = _mean_ci(
            [float(row["persistent_precursor_hit"]) for row in values]
        )
        recross = _mean_ci([row["tt_recross_fraction"] for row in values])
        q50 = kaplan_meier_median(
            [row["persistent_precursor_hit_event"] for row in values],
            censor_event=steps,
        )
        acceleration = None
        acceleration_lower_bound = False
        if q50 is not None and q50 > 0:
            if reference_q50 is None:
                acceleration = float(steps / q50)
                acceleration_lower_bound = True
            else:
                acceleration = float(reference_q50 / q50)
        rows.append(
            {
                "method": method,
                "n_episodes": len(values),
                "persistent_transport_auc": transport[0],
                "persistent_transport_auc_ci_low": transport[1],
                "persistent_transport_auc_ci_high": transport[2],
                "persistent_transport_auc_normalized": float(
                    transport[0] / max(reference_transport, 1.0e-12)
                ),
                "persistent_precursor_auc": precursor[0],
                "persistent_precursor_auc_ci_low": precursor[1],
                "persistent_precursor_auc_ci_high": precursor[2],
                "persistent_precursor_auc_normalized": float(
                    precursor[0] / max(reference_precursor, 1.0e-12)
                ),
                "persistent_precursor_hit_rate": hit_rate[0],
                "persistent_precursor_hit_rate_ci_low": max(hit_rate[1], 0.0),
                "persistent_precursor_hit_rate_ci_high": min(hit_rate[2], 1.0),
                "precursor_q50_events": q50,
                "precursor_acceleration_factor": acceleration,
                "precursor_acceleration_is_lower_bound": acceleration_lower_bound,
                "tt_recross_fraction": recross[0],
                "tt_recross_fraction_ci_low": max(recross[1], 0.0),
                "tt_recross_fraction_ci_high": min(recross[2], 1.0),
                "dynamics_error": None,
                "structure_error": None,
                "clock_error": None,
                "clock_calibrated": False,
            }
        )
    return rows

