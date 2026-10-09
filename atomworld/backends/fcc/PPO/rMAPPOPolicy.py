import torch
from torch.distributions import Categorical

from atomworld.backends.fcc.PPO.r_actor_critic import R_Actor, R_Critic
from atomworld.backends.fcc.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    def __init__(self, args, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.act_space = act_space
        self.num_agents = int(args.lattice_v_nums)
        if int(act_space.n) % self.num_agents != 0:
            raise ValueError(
                f"Global action count {act_space.n} is not divisible by vacancy count {self.num_agents}"
            )
        self.events_per_agent = int(act_space.n) // self.num_agents
        args.events_per_agent = self.events_per_agent

        base_scale = float(getattr(args, "policy_logit_scale", 0.2))
        start_scale = getattr(args, "policy_logit_scale_start", None)
        end_scale = getattr(args, "policy_logit_scale_end", None)
        self.policy_logit_scale_start = float(start_scale) if start_scale is not None else base_scale
        self.policy_logit_scale_end = float(end_scale) if end_scale is not None else self.policy_logit_scale_start
        self.policy_logit_scale_schedule = str(getattr(args, "policy_logit_scale_schedule", "constant")).strip().lower()
        self.policy_logit_scale = float(self.policy_logit_scale_start)

        rate_base_scale = float(getattr(args, "rate_prior_scale", 1.0))
        rate_start = getattr(args, "rate_prior_scale_start", None)
        rate_end = getattr(args, "rate_prior_scale_end", None)
        self.rate_prior_scale_start = float(rate_start) if rate_start is not None else rate_base_scale
        self.rate_prior_scale_end = float(rate_end) if rate_end is not None else self.rate_prior_scale_start
        self.rate_prior_scale_schedule = str(getattr(args, "rate_prior_scale_schedule", "constant")).strip().lower()
        self.rate_prior_scale = float(self.rate_prior_scale_start)

        energy_base_scale = float(getattr(args, "energy_prior_scale", 0.0))
        energy_start = getattr(args, "energy_prior_scale_start", None)
        energy_end = getattr(args, "energy_prior_scale_end", None)
        self.energy_prior_scale_start = float(energy_start) if energy_start is not None else energy_base_scale
        self.energy_prior_scale_end = float(energy_end) if energy_end is not None else self.energy_prior_scale_start
        self.energy_prior_scale_schedule = str(getattr(args, "energy_prior_scale_schedule", "constant")).strip().lower()
        self.energy_prior_scale = float(self.energy_prior_scale_start)
        self.energy_prior_downhill_bonus = float(getattr(args, "energy_prior_downhill_bonus", 0.20))
        self.energy_prior_solute_bonus = float(
            getattr(args, "energy_prior_solute_bonus", getattr(args, "energy_prior_cu_bonus", 0.10))
        )
        self.energy_prior_matrix_penalty = float(
            getattr(args, "energy_prior_matrix_penalty", getattr(args, "energy_prior_fe_penalty", 0.04))
        )
        self.energy_prior_clip = max(0.0, float(getattr(args, "energy_prior_clip", 3.0)))

        self.actor = R_Actor(args, self.act_space, self.device)
        self.critic = R_Critic(args, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _scheduled_value(self, start, end, schedule, episode, episodes):
        total_episodes = max(int(episodes), 1)
        if total_episodes <= 1:
            progress = 1.0
        else:
            progress = min(max(float(episode) / float(total_episodes - 1), 0.0), 1.0)

        if schedule == "constant":
            value = start
        elif schedule == "linear":
            value = start + progress * (end - start)
        elif schedule == "cosine":
            value = end + 0.5 * (start - end) * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
        else:
            raise ValueError(f"Unsupported schedule={schedule}")
        return float(value)

    def update_policy_logit_scale(self, episode, episodes):
        self.policy_logit_scale = self._scheduled_value(
            float(self.policy_logit_scale_start),
            float(self.policy_logit_scale_end),
            self.policy_logit_scale_schedule,
            episode,
            episodes,
        )
        self.rate_prior_scale = self._scheduled_value(
            float(self.rate_prior_scale_start),
            float(self.rate_prior_scale_end),
            self.rate_prior_scale_schedule,
            episode,
            episodes,
        )
        self.energy_prior_scale = self._scheduled_value(
            float(self.energy_prior_scale_start),
            float(self.energy_prior_scale_end),
            self.energy_prior_scale_schedule,
            episode,
            episodes,
        )
        self.args.policy_logit_scale = float(self.policy_logit_scale)
        self.args.rate_prior_scale = float(self.rate_prior_scale)
        self.args.energy_prior_scale = float(self.energy_prior_scale)
        return self.policy_logit_scale

    def _masked_standardize(self, values, legal_mask):
        legal_float = legal_mask.float()
        denom = legal_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (values * legal_float).sum(dim=-1, keepdim=True) / denom
        centered = values - mean
        var = (centered.pow(2) * legal_float).sum(dim=-1, keepdim=True) / denom
        std = torch.sqrt(var + 1e-6)
        return centered / std

    def _physics_energy_prior(self, event_features, legal_mask):
        if event_features is None or self.energy_prior_scale == 0.0:
            return 0.0

        event_features = torch.as_tensor(event_features, dtype=torch.float32, device=self.device)
        energy_gain = event_features[..., 4]
        downhill = event_features[..., 7]
        solute_bonus = event_features[..., 2] + event_features[..., 3]
        matrix_penalty = event_features[..., 1]

        original_shape = energy_gain.shape
        flat_energy_gain = energy_gain.flatten(start_dim=-2)
        flat_legal_mask = legal_mask.flatten(start_dim=-2)
        prior = self._masked_standardize(flat_energy_gain, flat_legal_mask).reshape(original_shape)
        if self.energy_prior_clip > 0.0:
            prior = torch.clamp(prior, -self.energy_prior_clip, self.energy_prior_clip)
        prior = (
            prior
            + self.energy_prior_downhill_bonus * downhill
            + self.energy_prior_solute_bonus * solute_bonus
            - self.energy_prior_matrix_penalty * matrix_penalty
        )
        return prior.masked_fill(~legal_mask, 0.0)

    def _compose_hazard_model(self, local_logits, base_rates, event_features=None):
        """
        Interpret the fused event logits as a compositional hazard model:
            lambda_e = exp(log_hazard_e),  Lambda = sum_e lambda_e
        so event probabilities and waiting-time prediction share the same state.
        """
        local_logits = torch.as_tensor(local_logits, dtype=torch.float32, device=self.device)
        base_rates = torch.as_tensor(base_rates, dtype=torch.float32, device=self.device)

        if local_logits.shape != base_rates.shape:
            raise ValueError(
                f"Shape mismatch between local_logits {tuple(local_logits.shape)} and base_rates {tuple(base_rates.shape)}"
            )
        if local_logits.dim() < 2 or local_logits.shape[-1] != self.events_per_agent:
            raise ValueError(
                "Expected local logits with trailing shape "
                f"(*, N_vac, {self.events_per_agent}), got {tuple(local_logits.shape)}"
            )

        legal_mask = base_rates > 0.0
        energy_prior = self._physics_energy_prior(event_features, legal_mask)
        log_hazard = (
            self.policy_logit_scale * local_logits
            + self.rate_prior_scale * torch.log(base_rates + 1e-12)
            + self.energy_prior_scale * energy_prior
        )
        log_hazard = log_hazard.masked_fill(~legal_mask, -1e10)

        hazard = torch.zeros_like(log_hazard)
        hazard = torch.where(legal_mask, torch.exp(log_hazard), hazard)

        hazard_flat = hazard.flatten(start_dim=-2)
        legal_flat = legal_mask.flatten(start_dim=-2)
        event_logits = log_hazard.flatten(start_dim=-2).masked_fill(~legal_flat, -1e10)
        total_hazard = hazard_flat.sum(dim=-1)

        no_legal = ~legal_flat.any(dim=-1)
        if bool(no_legal.any()):
            raise RuntimeError(f"No legal event available for {int(no_legal.sum().item())} sample(s): all base rates are zero.")

        expected_dt = total_hazard.clamp(min=1e-12).reciprocal()
        return {
            "event_logits": event_logits,
            "log_hazard": log_hazard,
            "hazard": hazard,
            "total_hazard": total_hazard,
            "expected_dt": expected_dt,
            "legal_mask": legal_mask,
            "energy_prior": energy_prior,
        }

    def _flatten_single_step_hazard_outputs(self, hazard_outputs):
        flat_outputs = {}
        for key, value in hazard_outputs.items():
            if not torch.is_tensor(value):
                flat_outputs[key] = value
                continue
            if key == "event_logits":
                flat_outputs[key] = value
            elif value.dim() == 0:
                flat_outputs[key] = value.reshape(1)
            else:
                flat_outputs[key] = value
        return flat_outputs

    def _flatten_time_batch_hazard_outputs(self, hazard_outputs, T, B):
        flat_outputs = {}
        for key, value in hazard_outputs.items():
            if not torch.is_tensor(value):
                flat_outputs[key] = value
                continue
            if key == "event_logits":
                flat_outputs[key] = value.reshape(T * B, value.shape[-1])
            elif value.dim() >= 3:
                flat_outputs[key] = value.reshape(T * B, *value.shape[2:])
            elif value.dim() == 2:
                flat_outputs[key] = value.reshape(T * B)
            else:
                flat_outputs[key] = value.reshape(T * B)
        return flat_outputs

    def _summarize_hazard_outputs(self, hazard_outputs, action):
        flat_hazard = hazard_outputs["hazard"].flatten(start_dim=-2)
        total_hazard = hazard_outputs["total_hazard"]
        expected_dt = hazard_outputs["expected_dt"]
        action = torch.as_tensor(action, dtype=torch.long, device=self.device).reshape(-1)

        if flat_hazard.dim() == 1:
            selected_hazard = flat_hazard[action[0]].reshape(1)
            total_hazard = total_hazard.reshape(1)
            expected_dt = expected_dt.reshape(1)
        else:
            selected_hazard = flat_hazard.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return {
            "pred_total_hazard": total_hazard.detach().cpu(),
            "pred_expected_dt": expected_dt.detach().cpu(),
            "pred_selected_hazard": selected_hazard.detach().cpu(),
        }

    def _combine_event_logits_batched(self, local_logits, base_rates, event_features=None):
        return self._compose_hazard_model(local_logits, base_rates, event_features=event_features)["event_logits"]

    def _combine_event_logits(self, local_logits, base_rates, event_features=None):
        event_logits = self._combine_event_logits_batched(local_logits, base_rates, event_features=event_features)
        if event_logits.dim() != 1:
            raise ValueError(f"Expected 1D event logits for a single sample, got shape {tuple(event_logits.shape)}")
        return event_logits

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        base_rates,
        event_features,
        sparse_defect_tokens=None,
        sparse_defect_masks=None,
        deterministic=False,
    ):
        local_logits, _, new_rnn_states_actor = self.actor.get_local_logits(
            obs,
            rnn_states_actor,
            masks,
            event_features=event_features,
            sparse_defect_tokens=sparse_defect_tokens,
            sparse_defect_masks=sparse_defect_masks,
        )
        hazard_outputs = self._flatten_single_step_hazard_outputs(
            self._compose_hazard_model(local_logits, base_rates, event_features=event_features)
        )
        dist = Categorical(logits=hazard_outputs["event_logits"])
        action = torch.argmax(hazard_outputs["event_logits"], dim=-1) if deterministic else dist.sample()
        action_log_probs = dist.log_prob(action).unsqueeze(-1)
        values = self.critic(share_obs, rnn_states_critic, masks, base_rates=base_rates, event_features=event_features)
        hazard_summary = self._summarize_hazard_outputs(hazard_outputs, action)
        return values, action.view(1, 1), action_log_probs.view(1, 1), new_rnn_states_actor, rnn_states_critic, hazard_summary

    def get_values(self, share_obs, rnn_states_critic=None, masks=None, base_rates=None, event_features=None):
        return self.critic(share_obs, rnn_states_critic, masks, base_rates=base_rates, event_features=event_features)

    def evaluate_actions(
        self,
        share_obs,
        obs,
        sparse_defect_tokens,
        sparse_defect_masks,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        base_rates,
        event_features,
    ):
        share_obs = torch.as_tensor(share_obs, dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        base_rates = torch.as_tensor(base_rates, dtype=torch.float32, device=self.device)
        event_features = torch.as_tensor(event_features, dtype=torch.float32, device=self.device)
        rnn_states_actor = torch.as_tensor(rnn_states_actor, dtype=torch.float32, device=self.device)
        sparse_defect_tokens = torch.as_tensor(sparse_defect_tokens, dtype=torch.float32, device=self.device)
        sparse_defect_masks = torch.as_tensor(sparse_defect_masks, dtype=torch.float32, device=self.device)

        if obs.dim() == 3:
            obs = obs.unsqueeze(1)
            share_obs = share_obs.unsqueeze(1)
            base_rates = base_rates.unsqueeze(1)
            event_features = event_features.unsqueeze(1)
            sparse_defect_tokens = sparse_defect_tokens.unsqueeze(1)
            sparse_defect_masks = sparse_defect_masks.unsqueeze(1)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            elif action.dim() == 1:
                action = action.view(-1, 1, 1)
            if masks.dim() == 2:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 1:
                masks = masks.view(-1, 1, 1)
            if rnn_states_actor.dim() == 3:
                rnn_states_actor = rnn_states_actor.unsqueeze(0)
        elif obs.dim() != 4:
            raise ValueError(f"Unsupported obs shape for evaluate_actions: {tuple(obs.shape)}")

        if action.dim() == 2:
            action = action.unsqueeze(-1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)

        T, B, N, _ = obs.shape
        if share_obs.dim() == 4:
            flat_share_obs = share_obs.reshape(T * B, N, share_obs.shape[-1])
        else:
            flat_share_obs = share_obs.reshape(T * B, share_obs.shape[-1])
        flat_base_rates = base_rates.reshape(T * B, N, self.events_per_agent)
        flat_event_features = event_features.reshape(
            T * B, N, self.events_per_agent, event_features.shape[-1]
        )
        values = self.critic(flat_share_obs, rnn_states_critic, None, base_rates=flat_base_rates, event_features=flat_event_features)

        local_logits_seq, _, _ = self.actor.get_local_logits(
            obs,
            rnn_states_actor,
            masks,
            event_features=event_features,
            sparse_defect_tokens=sparse_defect_tokens,
            sparse_defect_masks=sparse_defect_masks,
        )
        hazard_outputs = self._compose_hazard_model(local_logits_seq, base_rates, event_features=event_features)
        flat_hazard_outputs = self._flatten_time_batch_hazard_outputs(hazard_outputs, T, B)
        dist = Categorical(logits=flat_hazard_outputs["event_logits"])

        action = action.reshape(T, B, -1)
        if action.shape[-1] != 1:
            raise ValueError(f"Expected exactly one global action per timestep, got shape {tuple(action.shape)}")
        chosen_actions = action[..., 0].reshape(T * B)

        action_log_probs = dist.log_prob(chosen_actions).reshape(T * B, 1)
        dist_entropy = dist.entropy().mean()
        return values.reshape(T * B, 1), action_log_probs, dist_entropy, flat_hazard_outputs

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        base_rates,
        event_features,
        sparse_defect_tokens=None,
        sparse_defect_masks=None,
        deterministic=False,
        return_hazard_summary=False,
    ):
        local_logits, _, new_rnn_states_actor = self.actor.get_local_logits(
            obs,
            rnn_states_actor,
            masks,
            event_features=event_features,
            sparse_defect_tokens=sparse_defect_tokens,
            sparse_defect_masks=sparse_defect_masks,
        )
        hazard_outputs = self._flatten_single_step_hazard_outputs(
            self._compose_hazard_model(local_logits, base_rates, event_features=event_features)
        )
        dist = Categorical(logits=hazard_outputs["event_logits"])
        action = torch.argmax(hazard_outputs["event_logits"], dim=-1) if deterministic else dist.sample()
        if return_hazard_summary:
            return action.view(1, 1), new_rnn_states_actor, self._summarize_hazard_outputs(hazard_outputs, action)
        return action.view(1, 1), new_rnn_states_actor
