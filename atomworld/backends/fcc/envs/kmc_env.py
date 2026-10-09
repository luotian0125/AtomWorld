from atomworld.backends.fcc.envs.kmc import KMCEnv
import random
import numpy as np
try:
    from gymnasium import spaces
except ImportError:  # Compatibility with the original AtomWorld environment.
    from gym import spaces


class KMCEnvWrap(object):
    """
    Thin wrapper only.

    The physics kernel and RL env logic remain in atomworld.backends.fcc.envs.kmc.KMCEnv.
    This file only exposes gym-like spaces and forwards reset/step calls.
    """

    def __init__(self, args):
        self.env = KMCEnv(args)
        self.env.vacancy_pos = np.asarray(self.env.vacancy_pos)
        self.num_agents = args.lattice_v_nums

        self.history_len = int(getattr(args, "history_len", getattr(self.env, "history_len", 5)))
        self.events_per_agent = int(self.env.events_per_agent)
        self.spatial_obs_dim = int(self.env.spatial_obs_dim)
        self.history_obs_dim = self.history_len
        self.obs_dim = self.spatial_obs_dim + self.history_obs_dim
        self.share_obs_dim = self.obs_dim * self.num_agents + 9

        # Single global event space: event_id = vac_idx * K + dir_idx.
        self.action_space = spaces.Discrete(self.num_agents * self.events_per_agent)

        # States remain vacancy-wise because the actor still encodes all vacancies.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.obs_dim),
            dtype=np.float32,
        )
        self.share_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.share_obs_dim),
            dtype=np.float32,
        )

    def reset(self):
        return self.env.reset()

    def step(self, action, episode=None):
        obs, share_obs, positions, reward, done, info = self.env.step(action, episode)
        if 'total_energy' not in info and len(self.env.energy_history) > 0:
            info['total_energy'] = self.env.energy_history[-1]
        return obs, share_obs, positions, reward, done, info

    def seed(self, seed=None):
        seed = 1 if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)

    def get_system_stats(self):
        return self.env.get_system_stats()

    def get_base_rates(self):
        return self.env.get_base_rates()

    def get_base_rates_np(self, copy=True):
        return self.env.get_base_rates_np(copy=copy)

    def get_sparse_defect_context(self):
        return self.env.get_sparse_defect_context()

    def get_sparse_defect_context_np(self, copy=True):
        return self.env.get_sparse_defect_context(copy=copy)

    def get_event_features(self):
        return self.env.get_event_features_tensor()

    def get_event_features_np(self, copy=True):
        return self.env.get_event_features(copy=copy)

    def get_state_dict(self):
        return self.env.get_state_dict()

    def set_state_dict(self, state):
        self.env.set_state_dict(state)
