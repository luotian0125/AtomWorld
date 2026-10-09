"""Auditable, clockless degradation-surrogate reward for HCP training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from atomworld.backends.hcp.analysis.clusters import hydrogen_cluster_metrics


@dataclass(frozen=True)
class HCPRewardConfig:
    energy_drop_weight_per_eV: float = 1.0
    attractive_pair_weight: float = 0.25
    largest_cluster_weight: float = 1.0
    nonisolated_fraction_weight: float = 0.5
    trapped_fraction_weight: float = 1.0
    immediate_reverse_penalty: float = 0.02
    keyframe_bonus: float = 0.0

    def metadata(self):
        return asdict(self)


class HCPRewardModel:
    """Reward changes in dissolved-H precursor state, never elapsed time.

    No term labels a configuration as a hydride.  Pair connectivity and trap
    occupation are explicitly surrogate order parameters on the fixed alpha-Zr
    host.  The default keyframe bonus is zero so the policy is not rewarded for
    an event merely because the physics code classified it as rare.
    """

    def __init__(self, config=None):
        self.config = config or HCPRewardConfig()
        if not isinstance(self.config, HCPRewardConfig):
            raise TypeError("config must be an HCPRewardConfig")

    @staticmethod
    def _attractive_pair_count(kmc, metrics):
        attractive_labels = {
            shell.label
            for shell in kmc.energy_model.pair_shells
            if shell.interaction_energy_eV < 0.0
        }
        return int(
            sum(
                count
                for label, count in metrics["pair_counts"].items()
                if label in attractive_labels
            )
        )

    def snapshot(self, kmc):
        metrics = hydrogen_cluster_metrics(kmc.lattice, kmc.energy_model)
        n_hydrogen = max(int(kmc.num_agents), 1)
        trapped = sum(
            kmc.energy_model.trap_energy_eV(site) < 0.0
            for site in kmc.lattice.hydrogen_sites
        )
        attractive_pairs = self._attractive_pair_count(kmc, metrics)
        components = {
            "negative_energy_eV": -float(kmc.energy_eV),
            "attractive_pairs_per_h": attractive_pairs / n_hydrogen,
            "largest_cluster_fraction": float(
                metrics["largest_attractive_cluster_fraction"]
            ),
            "nonisolated_fraction": 1.0 - float(metrics["isolated_hydrogen_fraction"]),
            "trapped_fraction": trapped / n_hydrogen,
        }
        components["objective"] = self.objective_from_components(components)
        return components

    def objective_from_components(self, components):
        cfg = self.config
        return float(
            cfg.energy_drop_weight_per_eV * components["negative_energy_eV"]
            + cfg.attractive_pair_weight * components["attractive_pairs_per_h"]
            + cfg.largest_cluster_weight * components["largest_cluster_fraction"]
            + cfg.nonisolated_fraction_weight * components["nonisolated_fraction"]
            + cfg.trapped_fraction_weight * components["trapped_fraction"]
        )

    def transition_reward(self, before, after, policy_transition):
        cfg = self.config
        objective_delta = float(after["objective"] - before["objective"])
        reverse = bool(
            policy_transition.physical_transition.info.get("immediate_reverse", False)
        )
        keyframe = bool(policy_transition.keyframe.is_keyframe)
        reward = (
            objective_delta
            - cfg.immediate_reverse_penalty * float(reverse)
            + cfg.keyframe_bonus * float(keyframe)
        )
        return float(reward), {
            "objective_delta": objective_delta,
            "immediate_reverse_penalty": -cfg.immediate_reverse_penalty
            * float(reverse),
            "keyframe_bonus": cfg.keyframe_bonus * float(keyframe),
            "reward": float(reward),
        }
