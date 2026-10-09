import numpy as np


def _split_indices(total_length, chunk_length, num_mini_batch):
    total_length = int(total_length)
    chunk_length = max(1, min(int(chunk_length), total_length))
    starts = list(range(0, max(total_length - chunk_length + 1, 1), chunk_length))
    tail_start = max(0, total_length - chunk_length)
    if not starts:
        starts = [tail_start]
    elif starts[-1] != tail_start:
        starts.append(tail_start)
    perm = np.random.permutation(len(starts))
    groups = np.array_split(perm, max(1, min(num_mini_batch, len(starts))))
    return starts, groups, chunk_length


class SeparatedReplayBuffer(object):
    def __init__(self, args, act_space, obs_space=None, share_obs_space=None):
        self.episode_length = args.episode_length
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.num_agents = args.lattice_v_nums
        self.sparse_defect_max_tokens = int(getattr(args, "sparse_defect_max_tokens", 24))
        self.sparse_defect_feature_dim = 6
        self.event_feature_dim = max(9, int(getattr(args, "event_feature_dim", 9)))

        self.history_len = int(getattr(args, "history_len", 5))
        self.events_per_agent = int(getattr(act_space, "n", self.num_agents * 12)) // self.num_agents
        obs_shape = getattr(obs_space, "shape", None)
        if obs_shape is None:
            self.spatial_obs_dim = int(getattr(args, "spatial_obs_dim", 54))
            self.obs_dim = self.spatial_obs_dim + self.history_len
            obs_shape = (self.num_agents, self.obs_dim)
        else:
            self.obs_dim = int(obs_shape[-1])
            self.spatial_obs_dim = self.obs_dim - self.history_len
        share_obs_full_shape = getattr(share_obs_space, "shape", (self.num_agents, 9 + self.obs_dim * self.num_agents))
        share_obs_shape = (int(share_obs_full_shape[-1]),)

        self.obs = np.zeros((self.episode_length + 1, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, *share_obs_shape), dtype=np.float32)
        self.positions = np.zeros((self.episode_length + 1, self.num_agents, 3), dtype=np.int32)
        self.base_rates = np.zeros(
            (self.episode_length + 1, self.num_agents, self.events_per_agent), dtype=np.float32
        )
        self.event_features = np.zeros(
            (self.episode_length + 1, self.num_agents, self.events_per_agent, self.event_feature_dim),
            dtype=np.float32,
        )
        self.sparse_defect_tokens = np.zeros((self.episode_length + 1, self.num_agents, self.sparse_defect_max_tokens, self.sparse_defect_feature_dim), dtype=np.float32)
        self.sparse_defect_tokens[..., 0] = 2.0
        self.sparse_defect_masks = np.zeros((self.episode_length + 1, self.num_agents, self.sparse_defect_max_tokens), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros((self.episode_length + 1, 1, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.actions = np.zeros((self.episode_length, 1), dtype=np.int64)
        self.action_log_probs = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.kmc_delta_t = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length + 1, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, 1), dtype=np.float32)
        self.bad_masks = np.ones((self.episode_length + 1, 1), dtype=np.float32)
        self.active_masks = np.ones((self.episode_length + 1, 1), dtype=np.float32)
        self.step = 0
        self.valid_length = 0

    def reset_rollout(self):
        self.step = 0
        self.valid_length = 0
        self.rnn_states.fill(0.0)
        self.rnn_states_critic.fill(0.0)
        self.masks.fill(1.0)
        self.bad_masks.fill(1.0)
        self.active_masks.fill(1.0)
        self.sparse_defect_tokens.fill(0.0)
        self.sparse_defect_tokens[..., 0] = 2.0
        self.event_features.fill(0.0)
        self.kmc_delta_t.fill(0.0)
        self.sparse_defect_masks.fill(0.0)

    def insert(self, share_obs, obs, positions, base_rates, event_features, sparse_defect_tokens, sparse_defect_masks, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds, rewards, kmc_delta_t, masks, bad_masks=None, active_masks=None):
        share_obs_arr = np.asarray(share_obs, dtype=np.float32)
        if share_obs_arr.ndim > 1:
            share_obs_arr = share_obs_arr[0]
        self.share_obs[self.step + 1] = share_obs_arr
        self.obs[self.step + 1] = np.asarray(obs, dtype=np.float32)
        self.positions[self.step + 1] = np.asarray(positions, dtype=np.int32)
        self.base_rates[self.step + 1] = np.asarray(base_rates, dtype=np.float32)
        self.event_features[self.step + 1] = np.asarray(event_features, dtype=np.float32)
        self.sparse_defect_tokens[self.step + 1] = np.asarray(sparse_defect_tokens, dtype=np.float32)
        self.sparse_defect_masks[self.step + 1] = np.asarray(sparse_defect_masks, dtype=np.float32)
        self.rnn_states[self.step + 1] = np.asarray(rnn_states, dtype=np.float32)
        self.rnn_states_critic[self.step + 1] = np.asarray(rnn_states_critic, dtype=np.float32)
        self.masks[self.step + 1] = np.asarray(masks, dtype=np.float32)
        self.actions[self.step] = np.asarray(actions, dtype=np.int64).reshape(1)
        self.action_log_probs[self.step] = np.asarray(action_log_probs, dtype=np.float32).reshape(1)
        self.value_preds[self.step] = np.asarray(value_preds, dtype=np.float32).reshape(1)
        self.rewards[self.step] = np.asarray(rewards, dtype=np.float32).reshape(1)
        self.kmc_delta_t[self.step] = np.asarray(kmc_delta_t, dtype=np.float32).reshape(1)
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = np.asarray(bad_masks, dtype=np.float32)
        if active_masks is not None:
            self.active_masks[self.step + 1] = np.asarray(active_masks, dtype=np.float32)
        self.valid_length = max(self.valid_length, self.step + 1)
        self.step = (self.step + 1) % self.episode_length

    def finalize_trajectory(self):
        tail_start = self.valid_length
        if tail_start < self.episode_length:
            self.rewards[tail_start:] = 0.0
            self.actions[tail_start:] = 0
            self.action_log_probs[tail_start:] = 0.0
            self.kmc_delta_t[tail_start:] = 0.0
            self.value_preds[tail_start + 1 :] = 0.0
            self.returns[tail_start + 1 :] = 0.0
            self.masks[tail_start + 1 :] = 0.0
            self.bad_masks[tail_start + 1 :] = 0.0
            self.active_masks[tail_start + 1 :] = 0.0
            self.rnn_states[tail_start + 1 :] = 0.0
            self.rnn_states_critic[tail_start + 1 :] = 0.0
            self.event_features[tail_start + 1 :] = 0.0
            self.sparse_defect_tokens[tail_start + 1 :] = 0.0
            self.sparse_defect_tokens[tail_start + 1 :, ..., 0] = 2.0
            self.sparse_defect_masks[tail_start + 1 :] = 0.0

    def after_update(self):
        src_idx = self.valid_length
        self.share_obs[0] = self.share_obs[src_idx].copy()
        self.obs[0] = self.obs[src_idx].copy()
        self.positions[0] = self.positions[src_idx].copy()
        self.base_rates[0] = self.base_rates[src_idx].copy()
        self.event_features[0] = self.event_features[src_idx].copy()
        self.sparse_defect_tokens[0] = self.sparse_defect_tokens[src_idx].copy()
        self.sparse_defect_masks[0] = self.sparse_defect_masks[src_idx].copy()
        self.rnn_states[0] = self.rnn_states[src_idx].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[src_idx].copy()
        self.masks[0] = self.masks[src_idx].copy()
        self.bad_masks[0] = self.bad_masks[src_idx].copy()
        self.active_masks[0] = self.active_masks[src_idx].copy()
        self.step = 0
        self.valid_length = 0

    def compute_returns(self, next_value, value_normalizer=None):
        T = int(self.valid_length)
        if T <= 0:
            self.value_preds[:] = 0.0
            self.returns[:] = 0.0
            return
        self.value_preds[T] = np.asarray(next_value, dtype=np.float32).reshape(1)
        self.value_preds[T + 1 :] = 0.0
        self.returns[T + 1 :] = 0.0
        if self._use_gae:
            gae = 0.0
            for step in reversed(range(T)):
                if self._use_popart or self._use_valuenorm:
                    next_val = value_normalizer.denormalize(self.value_preds[step + 1])
                    cur_val = value_normalizer.denormalize(self.value_preds[step])
                else:
                    next_val = self.value_preds[step + 1]
                    cur_val = self.value_preds[step]
                delta = self.rewards[step] + self.gamma * next_val * self.masks[step + 1] - cur_val
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                if self._use_proper_time_limits:
                    gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + cur_val
            self.returns[T + 1 :] = 0.0
        else:
            self.returns[T] = np.asarray(next_value, dtype=np.float32).reshape(1)
            for step in reversed(range(T)):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
            self.returns[T + 1 :] = 0.0

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        T = int(self.valid_length)
        if T <= 0:
            return
        advantages = advantages[:T]
        starts, groups, L = _split_indices(T, data_chunk_length, num_mini_batch)
        max_chunk_batch_size = 8
        for group in groups:
            if len(group) == 0:
                continue
            group_starts = [starts[int(idx)] for idx in group]
            for offset in range(0, len(group_starts), max_chunk_batch_size):
                batch_starts = group_starts[offset:offset + max_chunk_batch_size]
                yield (
                    np.stack([self.share_obs[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.obs[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.sparse_defect_tokens[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.sparse_defect_masks[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.base_rates[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.event_features[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.rnn_states[start] for start in batch_starts], axis=0),
                    np.stack([self.rnn_states_critic[start] for start in batch_starts], axis=0),
                    np.stack([self.actions[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.value_preds[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.returns[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.kmc_delta_t[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.masks[start:start + L] for start in batch_starts], axis=1),
                    np.stack([self.action_log_probs[start:start + L] for start in batch_starts], axis=1),
                    np.stack([advantages[start:start + L] for start in batch_starts], axis=1),
                )
