"""Matched-seed, fixed-event evaluation for clockless HCP policies."""

from __future__ import annotations

import numpy as np
import torch

from atomworld.backends.hcp.learning.hcp_binding import HCPAtomWorldBinding


EVALUATION_METHODS = (
    "atomworld",
    "kmc_rate_sampled_clockless",
    "rate_greedy",
    "energy_greedy",
    "random_uniform",
)


def _baseline_event(catalog, method, rng):
    legal = catalog.legal_mask.reshape(-1)
    legal_ids = np.flatnonzero(legal)
    if legal_ids.size == 0:
        raise RuntimeError("evaluation state contains no legal event")
    rates = catalog.rates.reshape(-1)
    if method == "kmc_rate_sampled_clockless":
        weights = rates[legal_ids]
        return int(rng.choice(legal_ids, p=weights / weights.sum()))
    if method == "rate_greedy":
        return int(legal_ids[np.argmax(rates[legal_ids])])
    if method == "energy_greedy":
        delta_energy = catalog.features[..., 11].reshape(-1)
        best_delta = np.min(delta_energy[legal_ids])
        tied = legal_ids[np.isclose(delta_energy[legal_ids], best_delta)]
        return int(tied[np.argmax(rates[tied])])
    if method == "random_uniform":
        return int(rng.choice(legal_ids))
    raise ValueError(f"unknown baseline method {method!r}")


def _evaluate_episode(
    model,
    *,
    environment_factory,
    reward_model,
    method,
    seed,
    steps,
    device,
    model_config,
    policy_temperature,
):
    env = environment_factory(int(seed))
    rng = np.random.default_rng(int(seed) + 32452843)
    initial = reward_model.snapshot(env.kmc)
    total_reward = 0.0
    reverse_count = 0
    keyframe_count = 0
    previous_long_state = None
    update_mask = np.ones(env.num_agents, dtype=bool)
    binding = None
    if method == "atomworld":
        binding = HCPAtomWorldBinding(
            env, config=model_config, device=device, model=model
        )

    for _ in range(int(steps)):
        catalog = env.enumerate_events()
        if method == "atomworld":
            event_id, next_long_state = binding.select_event(
                previous_long_state=previous_long_state,
                update_mask=update_mask,
                deterministic=True,
                policy_temperature=policy_temperature,
            )
        else:
            event_id = _baseline_event(catalog, method, rng)
            next_long_state = None
        before = reward_model.snapshot(env.kmc)
        transition = env.execute_event(event_id)
        after = reward_model.snapshot(env.kmc)
        reward, _ = reward_model.transition_reward(before, after, transition)
        total_reward += reward
        reverse_count += int(
            transition.physical_transition.info.get("immediate_reverse", False)
        )
        keyframe_count += int(transition.keyframe.is_keyframe)
        if method == "atomworld":
            previous_long_state = next_long_state.detach()
            update_mask = transition.long_memory_update_mask
    final = reward_model.snapshot(env.kmc)
    return {
        "seed": int(seed),
        "events": int(steps),
        "return": float(total_reward),
        "initial_objective": float(initial["objective"]),
        "final_objective": float(final["objective"]),
        "objective_gain": float(final["objective"] - initial["objective"]),
        "reverse_fraction": reverse_count / max(int(steps), 1),
        "keyframe_fraction": keyframe_count / max(int(steps), 1),
        "final_attractive_pairs_per_h": float(final["attractive_pairs_per_h"]),
        "final_largest_cluster_fraction": float(final["largest_cluster_fraction"]),
        "final_nonisolated_fraction": float(final["nonisolated_fraction"]),
        "clock_owner": "none_fixed_event_budget",
    }


def evaluate_methods(
    model,
    *,
    environment_factory,
    reward_model,
    seeds,
    steps,
    device,
    model_config,
    policy_temperature=1.0,
    methods=EVALUATION_METHODS,
):
    methods = tuple(methods)
    unknown = set(methods) - set(EVALUATION_METHODS)
    if unknown:
        raise ValueError(f"unknown evaluation methods: {sorted(unknown)}")
    was_training = model.training
    model.eval()
    payload = {
        "clock_owner": "none_fixed_event_budget",
        "claim_boundary": (
            "fixed-event precursor-surrogate comparison; no physical-time or "
            "hydride-nucleation claim"
        ),
        "steps_per_episode": int(steps),
        "seeds": [int(seed) for seed in seeds],
        "methods": {},
    }
    with torch.no_grad():
        for method in methods:
            episodes = [
                _evaluate_episode(
                    model,
                    environment_factory=environment_factory,
                    reward_model=reward_model,
                    method=method,
                    seed=seed,
                    steps=steps,
                    device=device,
                    model_config=model_config,
                    policy_temperature=policy_temperature,
                )
                for seed in seeds
            ]
            summary = {}
            for key in (
                "return",
                "initial_objective",
                "final_objective",
                "objective_gain",
                "reverse_fraction",
                "keyframe_fraction",
                "final_attractive_pairs_per_h",
                "final_largest_cluster_fraction",
                "final_nonisolated_fraction",
            ):
                values = np.asarray([episode[key] for episode in episodes])
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(
                    np.std(values, ddof=1) if len(values) > 1 else 0.0
                )
            payload["methods"][method] = {
                "summary": summary,
                "episodes": episodes,
            }
    if "atomworld" in payload["methods"]:
        policy = payload["methods"]["atomworld"]["summary"]
        payload["atomworld_comparisons"] = {}
        for method, values in payload["methods"].items():
            if method == "atomworld":
                continue
            baseline = values["summary"]
            payload["atomworld_comparisons"][method] = {
                "delta_mean_return": policy["mean_return"] - baseline["mean_return"],
                "delta_mean_objective_gain": policy["mean_objective_gain"]
                - baseline["mean_objective_gain"],
            }
    model.train(was_training)
    return payload
