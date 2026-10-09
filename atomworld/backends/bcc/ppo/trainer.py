import numpy as np
import torch
import torch.nn as nn

from atomworld.backends.bcc.ppo.utils.util import check
from atomworld.backends.bcc.utils.util import get_gard_norm, huber_loss, mse_loss
from atomworld.backends.bcc.utils.valuenorm import ValueNorm


class R_MAPPO:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.aux_phys_teacher_coef = float(getattr(args, "aux_phys_teacher_coef", 0.0))
        self.aux_teacher_energy_weight = float(getattr(args, "aux_teacher_energy_weight", 1.0))
        self.aux_teacher_rate_weight = float(getattr(args, "aux_teacher_rate_weight", 0.35))
        self.aux_teacher_cu_bonus = float(getattr(args, "aux_teacher_cu_bonus", 0.15))
        self.aux_teacher_fe_penalty = float(getattr(args, "aux_teacher_fe_penalty", 0.05))
        self.aux_teacher_temperature = max(1e-4, float(getattr(args, "aux_teacher_temperature", 0.75)))
        self.aux_event_time_coef = float(getattr(args, "aux_event_time_coef", 0.0))

        assert not (self._use_popart and self._use_valuenorm)
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        value_loss = torch.max(value_loss_original, value_loss_clipped) if self._use_clipped_value_loss else value_loss_original
        return value_loss.mean()

    def _masked_standardize(self, values, mask):
        mask_f = mask.float()
        denom = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (values * mask_f).sum(dim=-1, keepdim=True) / denom
        centered = values - mean
        var = (centered.pow(2) * mask_f).sum(dim=-1, keepdim=True) / denom
        std = torch.sqrt(var + 1e-6)
        return centered / std

    def _physical_teacher_loss(self, event_logits, event_features):
        if self.aux_phys_teacher_coef <= 0.0:
            return torch.zeros((), dtype=event_logits.dtype, device=event_logits.device)

        event_features = torch.as_tensor(event_features, dtype=torch.float32, device=event_logits.device)
        flat_feat = event_features.reshape(event_logits.shape[0], event_logits.shape[1], event_features.shape[-1])
        legal_mask = flat_feat[..., 0] > 0.5
        no_legal = ~legal_mask.any(dim=-1)
        if bool(no_legal.any()):
            legal_mask = legal_mask.clone()
            legal_mask[no_legal] = True

        energy_gain = self._masked_standardize(flat_feat[..., 4], legal_mask)
        log_rate = self._masked_standardize(flat_feat[..., 6], legal_mask)
        cu_bonus = flat_feat[..., 2]
        fe_penalty = flat_feat[..., 1]

        teacher_scores = (
            self.aux_teacher_energy_weight * energy_gain
            + self.aux_teacher_rate_weight * log_rate
            + self.aux_teacher_cu_bonus * cu_bonus
            - self.aux_teacher_fe_penalty * fe_penalty
        )
        teacher_scores = teacher_scores.masked_fill(~legal_mask, -1e10)
        teacher_probs = torch.softmax(teacher_scores / self.aux_teacher_temperature, dim=-1)
        student_log_probs = torch.log_softmax(event_logits, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

    def _event_time_aux_loss(self, hazard_outputs, base_rates, kmc_delta_t):
        pred_event_logits = hazard_outputs["event_logits"]
        pred_total_hazard = hazard_outputs["total_hazard"].reshape(-1)
        pred_expected_dt = hazard_outputs["expected_dt"].reshape(-1)
        pred_hazard = hazard_outputs["hazard"]

        target_hazard = torch.as_tensor(base_rates, dtype=torch.float32, device=self.device).reshape_as(pred_hazard)
        target_hazard = torch.clamp(target_hazard, min=0.0)
        flat_target_hazard = target_hazard.flatten(start_dim=-2)
        target_total_hazard = flat_target_hazard.sum(dim=-1)

        valid_mask = target_total_hazard > 0.0
        zero = torch.zeros((), dtype=pred_event_logits.dtype, device=self.device)
        if not bool(valid_mask.any()):
            return zero, zero, zero, zero, zero

        target_event_probs = flat_target_hazard / target_total_hazard.unsqueeze(-1).clamp(min=1e-12)
        pred_log_event_probs = torch.log_softmax(pred_event_logits, dim=-1)
        teacher_cross_entropy = -(target_event_probs * pred_log_event_probs).sum(dim=-1)
        target_event_entropy = -(
            target_event_probs.clamp(min=1e-12) * torch.log(target_event_probs.clamp(min=1e-12))
        ).sum(dim=-1)
        event_kl = teacher_cross_entropy - target_event_entropy

        kmc_delta_t = torch.as_tensor(kmc_delta_t, dtype=torch.float32, device=self.device).reshape(-1)
        pred_time_nll = -torch.log(pred_total_hazard.clamp(min=1e-12)) + pred_total_hazard * kmc_delta_t
        target_time_nll = -torch.log(target_total_hazard.clamp(min=1e-12)) + target_total_hazard * kmc_delta_t
        time_nll_residual = pred_time_nll - target_time_nll

        pred_log_total = torch.log(pred_total_hazard.clamp(min=1e-12))
        target_log_total = torch.log(target_total_hazard.clamp(min=1e-12))
        log_total_hazard_mae = torch.abs(pred_log_total - target_log_total)

        target_expected_dt = target_total_hazard.clamp(min=1e-12).reciprocal()
        log_expected_dt_mae = torch.abs(
            torch.log(pred_expected_dt.clamp(min=1e-12)) - torch.log(target_expected_dt)
        )

        event_kl = event_kl[valid_mask]
        time_nll_residual = time_nll_residual[valid_mask]
        log_total_hazard_mae = log_total_hazard_mae[valid_mask]
        log_expected_dt_mae = log_expected_dt_mae[valid_mask]
        event_time_loss = (event_kl + time_nll_residual).mean()
        return (
            event_time_loss,
            event_kl.mean(),
            time_nll_residual.mean(),
            log_total_hazard_mae.mean(),
            log_expected_dt_mae.mean(),
        )

    def ppo_update(self, sample, update_actor=True):
        (
            share_obs_batch,
            obs_batch,
            sparse_defect_tokens_batch,
            sparse_defect_masks_batch,
            base_rates_batch,
            event_features_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            kmc_delta_t_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv).reshape(-1, 1)
        adv_targ = check(adv_targ).to(**self.tpdv).reshape(-1, 1)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv).reshape(-1, 1)
        return_batch = check(return_batch).to(**self.tpdv).reshape(-1, 1)

        values, action_log_probs, dist_entropy, hazard_outputs = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            sparse_defect_tokens_batch,
            sparse_defect_masks_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            base_rates_batch,
            event_features_batch,
        )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.min(surr1, surr2).mean()
        aux_phys_loss = self._physical_teacher_loss(hazard_outputs["event_logits"], event_features_batch)
        event_time_aux_loss, event_kl, time_nll_residual, log_total_hazard_mae, log_expected_dt_mae = self._event_time_aux_loss(
            hazard_outputs,
            base_rates_batch,
            kmc_delta_t_batch,
        )

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (
                policy_loss
                + self.aux_phys_teacher_coef * aux_phys_loss
                + self.aux_event_time_coef * event_time_aux_loss
                - dist_entropy * self.entropy_coef
            ).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm) if self._use_max_grad_norm else get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm) if self._use_max_grad_norm else get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            aux_phys_loss,
            event_time_aux_loss,
            event_kl,
            time_nll_residual,
            log_total_hazard_mae,
            log_expected_dt_mae,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer, update_actor=True):
        valid_length = int(buffer.valid_length)
        if valid_length <= 0:
            raise ValueError("Cannot train on an empty rollout")
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:valid_length] - self.value_normalizer.denormalize(buffer.value_preds[:valid_length])
        else:
            advantages = buffer.returns[:valid_length] - buffer.value_preds[:valid_length]
        mean_advantages = np.nanmean(advantages)
        std_advantages = np.nanstd(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "aux_phys_loss": 0.0,
            "aux_event_time_loss": 0.0,
            "event_kl": 0.0,
            "time_nll_residual": 0.0,
            "log_total_hazard_mae": 0.0,
            "log_expected_dt_mae": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }
        num_updates = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    aux_phys_loss,
                    aux_event_time_loss,
                    event_kl,
                    time_nll_residual,
                    log_total_hazard_mae,
                    log_expected_dt_mae,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["aux_phys_loss"] += aux_phys_loss.item()
                train_info["aux_event_time_loss"] += aux_event_time_loss.item()
                train_info["event_kl"] += event_kl.item()
                train_info["time_nll_residual"] += time_nll_residual.item()
                train_info["log_total_hazard_mae"] += log_total_hazard_mae.item()
                train_info["log_expected_dt_mae"] += log_expected_dt_mae.item()
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += imp_weights.mean().item()
                num_updates += 1

        num_updates = max(1, num_updates)
        for key in train_info:
            train_info[key] /= num_updates
        train_info["mean_advantages"] = float(mean_advantages)
        train_info["std_advantages"] = float(std_advantages)
        return train_info

    def prep_training(self):
        # PPO must recompute the same conditional distribution used at collection.
        # eval() disables fresh dropout masks; it does not disable autograd.
        self.policy.actor.eval()
        self.policy.critic.eval()
        # cuDNN requires a training-mode GRU to save backward intermediates.
        # These GRUs have zero dropout, so this preserves rollout probabilities.
        self.policy.actor.rnn.train()
        self.policy.critic.rnn.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
