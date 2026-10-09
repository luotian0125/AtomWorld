import os
import json
import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None
from tensorboardX import SummaryWriter

from atomworld.backends.bcc.utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        if self.use_wandb and wandb is None:
            raise RuntimeError("Weights & Biases was requested, but wandb is not installed. Use --no_wandb or install wandb.")
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.num_agents = self.all_args.lattice_v_nums

        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.model_dir = self.all_args.model_dir

        if not self.use_render:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                os.makedirs(self.log_dir, exist_ok=True)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                os.makedirs(self.save_dir, exist_ok=True)

        from atomworld.backends.bcc.ppo.trainer import R_MAPPO as TrainAlgo
        from atomworld.backends.bcc.ppo.policy import R_MAPPOPolicy as Policy

        self.policy = Policy(self.all_args, self.envs.action_space, device=self.device)
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        if self.model_dir is not None:
            if self.use_eval:
                self.restore4eval()
            else:
                self.restore()

        self.buffer = SeparatedReplayBuffer(
            self.all_args,
            self.envs.action_space,
            self.envs.observation_space,
            self.envs.share_observation_space,
        )

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_idx = int(getattr(self.buffer, "valid_length", self.buffer.episode_length))
        next_value = self.trainer.policy.get_values(
            self.buffer.share_obs[next_idx],
            self.buffer.rnn_states_critic[next_idx],
            self.buffer.masks[next_idx],
            self.buffer.base_rates[next_idx],
            self.buffer.event_features[next_idx],
            obs=self.buffer.obs[next_idx],
            rnn_states_actor=self.buffer.rnn_states[next_idx],
            sparse_defect_tokens=self.buffer.sparse_defect_tokens[next_idx],
            sparse_defect_masks=self.buffer.sparse_defect_masks[next_idx],
        )
        self.buffer.compute_returns(_t2n(next_value), self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    def save(self, save_dir=None):
        target_dir = str(save_dir) if save_dir is not None else self.save_dir
        os.makedirs(target_dir, exist_ok=True)
        torch.save(self.trainer.policy.actor.state_dict(), os.path.join(target_dir, "actor_agent.pt"))
        torch.save(self.trainer.policy.critic.state_dict(), os.path.join(target_dir, "critic_agent.pt"))
        if self.trainer._use_valuenorm:
            torch.save(self.trainer.value_normalizer.state_dict(), os.path.join(target_dir, "vnrom_agent.pt"))
        metadata = {
            "parameters": vars(self.all_args),
            "effective_scales": {
                "policy_logit_scale": self.policy.policy_logit_scale,
                "rate_prior_scale": self.policy.rate_prior_scale,
                "energy_prior_scale": self.policy.energy_prior_scale,
            },
        }
        with open(os.path.join(target_dir, "checkpoint_config.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)

    def restore(self):
        actor_path = os.path.join(str(self.model_dir), "actor_agent.pt")
        critic_path = os.path.join(str(self.model_dir), "critic_agent.pt")
        actor_state = torch.load(actor_path, map_location=self.device)
        critic_state = torch.load(critic_path, map_location=self.device)
        actor_msg = self.policy.actor.load_state_dict(actor_state, strict=False)
        critic_msg = self.policy.critic.load_state_dict(critic_state, strict=False)
        if actor_msg.missing_keys or actor_msg.unexpected_keys:
            print(f"[Restore] actor partial load. missing={list(actor_msg.missing_keys)} unexpected={list(actor_msg.unexpected_keys)}")
        if critic_msg.missing_keys or critic_msg.unexpected_keys:
            print(f"[Restore] critic partial load. missing={list(critic_msg.missing_keys)} unexpected={list(critic_msg.unexpected_keys)}")
        if self.trainer._use_valuenorm:
            vn_path = os.path.join(str(self.model_dir), "vnrom_agent.pt")
            if os.path.exists(vn_path):
                self.trainer.value_normalizer.load_state_dict(torch.load(vn_path, map_location=self.device))

    def restore4eval(self):
        actor_path = os.path.join(str(self.model_dir), "actor_agent.pt")
        actor_state = torch.load(actor_path, map_location=self.device)
        actor_msg = self.policy.actor.load_state_dict(actor_state, strict=False)
        if actor_msg.missing_keys or actor_msg.unexpected_keys:
            print(f"[RestoreEval] actor partial load. missing={list(actor_msg.missing_keys)} unexpected={list(actor_msg.unexpected_keys)}")

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos[0].items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars("reward", {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
