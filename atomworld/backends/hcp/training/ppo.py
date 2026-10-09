"""Clockless rollout collection and PPO updates for AtomWorld-HCP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class HCPRolloutBatch:
    model_inputs: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    base_rates: torch.Tensor
    metrics: dict[str, float]


def generalized_advantage_estimate(
    rewards, values, dones, bootstrap_value, *, gamma, gae_lambda
):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have identical shapes")
    advantages = np.zeros_like(rewards)
    next_value = float(bootstrap_value)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        accumulator = delta + gamma * gae_lambda * continuation * accumulator
        advantages[index] = accumulator
        next_value = values[index]
    return advantages, advantages + values


class ClocklessRolloutCollector:
    """Collect policy events while keeping both passive and policy clocks off."""

    def __init__(
        self,
        binding,
        *,
        environment_factory,
        reward_model,
        rollout_steps,
        episode_length,
        seed,
        gamma=0.99,
        gae_lambda=0.95,
        policy_temperature=1.0,
    ):
        self.binding = binding
        self.environment_factory = environment_factory
        self.reward_model = reward_model
        self.rollout_steps = int(rollout_steps)
        self.episode_length = int(episode_length)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.policy_temperature = float(policy_temperature)
        if self.rollout_steps <= 0 or self.episode_length <= 0:
            raise ValueError("rollout_steps and episode_length must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if self.policy_temperature <= 0.0:
            raise ValueError("policy_temperature must be positive")
        self.episode_index = 0
        self.episode_step = 0
        self.total_events = 0
        self.completed_episodes = 0
        self.previous_long_state = None
        self.update_mask = np.ones(self.binding.env.num_agents, dtype=bool)

    def _reset_episode(self):
        self.episode_index += 1
        self.binding.env = self.environment_factory(
            self.seed + 1009 * self.episode_index
        )
        self.previous_long_state = None
        self.update_mask = np.ones(self.binding.env.num_agents, dtype=bool)
        self.episode_step = 0

    @staticmethod
    def _stack_inputs(rows):
        return {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}

    def collect(self):
        model = self.binding.model
        model.eval()
        input_rows = []
        actions = []
        log_probabilities = []
        values = []
        rewards = []
        dones = []
        base_rates = []
        objective_deltas = []
        reverse_events = 0
        keyframe_events = 0
        episode_returns = []
        active_episode_return = 0.0

        for _ in range(self.rollout_steps):
            model_inputs = self.binding.model_inputs(
                previous_long_state=self.previous_long_state,
                update_mask=self.update_mask,
            )
            with torch.no_grad():
                output = model(
                    **model_inputs,
                    policy_temperature=self.policy_temperature,
                )
                action = output.distribution.sample()
                log_probability = output.distribution.log_prob(action)
            event_id = int(action.item())
            catalog = self.binding.env.enumerate_events()
            before = self.reward_model.snapshot(self.binding.env.kmc)
            transition = self.binding.env.execute_event(event_id)
            after = self.reward_model.snapshot(self.binding.env.kmc)
            reward, reward_info = self.reward_model.transition_reward(
                before, after, transition
            )

            self.episode_step += 1
            self.total_events += 1
            done = self.episode_step >= self.episode_length
            active_episode_return += reward

            input_rows.append(
                {key: value.detach().cpu() for key, value in model_inputs.items()}
            )
            actions.append(event_id)
            log_probabilities.append(float(log_probability.item()))
            values.append(float(output.value.item()))
            rewards.append(reward)
            dones.append(done)
            base_rates.append(catalog.rates.astype(np.float32, copy=True))
            objective_deltas.append(reward_info["objective_delta"])
            reverse_events += int(
                transition.physical_transition.info.get("immediate_reverse", False)
            )
            keyframe_events += int(transition.keyframe.is_keyframe)

            self.previous_long_state = output.long_memory.detach()
            self.update_mask = transition.long_memory_update_mask
            if done:
                episode_returns.append(active_episode_return)
                active_episode_return = 0.0
                self.completed_episodes += 1
                self._reset_episode()

        if dones[-1]:
            bootstrap_value = 0.0
        else:
            with torch.no_grad():
                bootstrap = self.binding.forward(
                    previous_long_state=self.previous_long_state,
                    update_mask=self.update_mask,
                )
            bootstrap_value = float(bootstrap.value.item())
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            dones,
            bootstrap_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        advantages = (advantages - advantage_mean) / max(advantage_std, 1.0e-8)
        metrics = {
            "rollout_reward_mean": float(np.mean(rewards)),
            "rollout_reward_sum": float(np.sum(rewards)),
            "rollout_objective_delta_sum": float(np.sum(objective_deltas)),
            "rollout_reverse_fraction": reverse_events / self.rollout_steps,
            "rollout_keyframe_fraction": keyframe_events / self.rollout_steps,
            "rollout_value_mean": float(np.mean(values)),
            "rollout_advantage_raw_mean": advantage_mean,
            "rollout_advantage_raw_std": advantage_std,
            "completed_episodes": float(self.completed_episodes),
            "total_events": float(self.total_events),
        }
        if episode_returns:
            metrics["completed_episode_return_mean"] = float(np.mean(episode_returns))
        return HCPRolloutBatch(
            model_inputs=self._stack_inputs(input_rows),
            actions=torch.as_tensor(actions, dtype=torch.long),
            old_log_probabilities=torch.as_tensor(
                log_probabilities, dtype=torch.float32
            ),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32),
            base_rates=torch.as_tensor(np.stack(base_rates), dtype=torch.float32),
            metrics=metrics,
        )


class HCPPPOUpdater:
    def __init__(
        self,
        model,
        actor_optimizer,
        critic_optimizer,
        *,
        device,
        ppo_epochs=4,
        minibatch_size=64,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        kmc_kl_coefficient=0.01,
        max_grad_norm=1.0,
    ):
        self.model = model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.device = torch.device(device)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.kmc_kl_coefficient = float(kmc_kl_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        if self.ppo_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("ppo_epochs and minibatch_size must be positive")

    def _slice_inputs(self, inputs, indices):
        return {
            key: value[indices].to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

    def update(self, batch):
        self.model.train()
        count = int(batch.actions.shape[0])
        metric_rows = []
        actor_parameters = (
            list(self.model.keyframes.parameters())
            + list(self.model.memory.parameters())
            + list(self.model.actor.parameters())
        )
        critic_parameters = list(self.model.critic.parameters())
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                inputs = self._slice_inputs(batch.model_inputs, indices)
                actions = batch.actions[indices].to(self.device)
                old_log_probs = batch.old_log_probabilities[indices].to(self.device)
                advantages = batch.advantages[indices].to(self.device)
                returns = batch.returns[indices].to(self.device)
                rates = batch.base_rates[indices].to(self.device)

                output = self.model(**inputs)
                log_probs = output.distribution.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                policy_loss = -torch.minimum(
                    ratio * advantages, clipped_ratio * advantages
                ).mean()
                entropy = output.distribution.entropy().mean()
                value_loss = F.mse_loss(output.value.reshape(-1), returns)

                kmc_kl = torch.zeros((), device=self.device)
                if self.kmc_kl_coefficient > 0.0:
                    flat_rates = rates.flatten(start_dim=-2).clamp_min(0.0)
                    target = flat_rates / flat_rates.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0e-30)
                    log_policy = torch.log_softmax(output.global_logits, dim=-1)
                    positive = target > 0.0
                    terms = torch.zeros_like(target)
                    terms[positive] = target[positive] * (
                        torch.log(target[positive]) - log_policy[positive]
                    )
                    kmc_kl = terms.sum(dim=-1).mean()

                actor_loss = (
                    policy_loss
                    - self.entropy_coefficient * entropy
                    + self.kmc_kl_coefficient * kmc_kl
                )
                total_loss = actor_loss + self.value_coefficient * value_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, self.max_grad_norm
                )
                critic_grad = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                approximate_kl = (old_log_probs - log_probs).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()
                )
                metric_rows.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "kmc_kl": float(kmc_kl.detach().cpu()),
                        "approximate_kl": float(approximate_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "actor_grad_norm": float(
                            torch.as_tensor(actor_grad).detach().cpu()
                        ),
                        "critic_grad_norm": float(
                            torch.as_tensor(critic_grad).detach().cpu()
                        ),
                    }
                )
        return {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
