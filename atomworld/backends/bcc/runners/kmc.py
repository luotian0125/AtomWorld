import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None
import matplotlib.pyplot as plt
from tqdm import tqdm

from atomworld.backends.bcc.runners.base import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


class KMCRunner(Runner):
    def __init__(self, args, config):
        super().__init__(config)
        self.reward_history = []
        self.energy_change_history = []
        self.abs_energy_history = []
        self.enable_episode_plots = bool(getattr(self.all_args, "enable_episode_plots", False))
        self.plot_interval = int(getattr(self.all_args, "plot_interval", 0))
        self.enable_snapshot_panels = bool(getattr(self.all_args, "enable_snapshot_panels", False))
        self.snapshot_panel_interval = max(1, int(getattr(self.all_args, "snapshot_panel_interval", 25)))
        snapshot_steps = getattr(self.all_args, "snapshot_panel_steps", None)
        self.snapshot_panel_steps = None if snapshot_steps is None else [int(x) for x in snapshot_steps]
        self.snapshot_panel_show_vacancy = bool(getattr(self.all_args, "snapshot_panel_show_vacancy", False))
        self.snapshot_panel_output_root = str(getattr(self.all_args, "snapshot_panel_output_root", "")).strip()
        self.show_collect_progress = bool(getattr(self.all_args, "show_collect_progress", False))
        self.show_eval_progress = bool(getattr(self.all_args, "show_eval_progress", False))

        self.enable_auto_compare = bool(getattr(self.all_args, "enable_auto_compare", False))
        self.auto_compare_interval = max(1, int(getattr(self.all_args, "auto_compare_interval", 25)))
        self.auto_compare_workers = int(getattr(self.all_args, "auto_compare_workers", 1))
        self.auto_compare_require_success = bool(getattr(self.all_args, "auto_compare_require_success", False))
        self.auto_compare_script = str(getattr(self.all_args, "auto_compare_script", "scripts/compare_methods.py"))
        self.auto_compare_python_bin = str(getattr(self.all_args, "auto_compare_python_bin", "")).strip() or sys.executable
        self.auto_compare_output_root = str(getattr(self.all_args, "auto_compare_output_root", "")).strip()
        self.auto_compare_bench_step = int(getattr(self.all_args, "auto_compare_bench_step", -1))
        self.auto_compare_energy_series_interval_steps = int(getattr(self.all_args, "auto_compare_energy_series_interval_steps", 50))
        self.auto_compare_plot_band = str(getattr(self.all_args, "auto_compare_plot_band", "stderr"))
        self.auto_compare_bootstrap_samples = int(getattr(self.all_args, "auto_compare_bootstrap_samples", 2000))
        self.auto_compare_seeds = [int(x) for x in getattr(self.all_args, "auto_compare_seeds", [1, 2, 3, 4, 5])]
        self.auto_compare_modes = [str(x) for x in getattr(self.all_args, "auto_compare_modes", ["traditional", "rl4kmc_stochastic", "rl4kmc_greedy"])]
        self.auto_compare_target_energy_drops = [float(x) for x in getattr(self.all_args, "auto_compare_target_energy_drops", [4.0, 5.0, 6.0])]
        self.auto_compare_rate_energy_beta = float(getattr(self.all_args, "auto_compare_rate_energy_beta", 1.0))
        self.enable_auto_paper_metrics = bool(getattr(self.all_args, "enable_auto_paper_metrics", False))
        self.auto_paper_metrics_script = str(getattr(self.all_args, "auto_paper_metrics_script", "scripts/build_paper_metrics.py"))
        self.auto_paper_metrics_reference_mode = str(getattr(self.all_args, "auto_paper_metrics_reference_mode", "traditional"))
        self.auto_paper_metrics_main_mode = str(getattr(self.all_args, "auto_paper_metrics_main_mode", "rl4kmc_stochastic"))
        self.auto_paper_metrics_method_label = str(getattr(self.all_args, "auto_paper_metrics_method_label", "AtomWorld-Mem"))
        self.auto_paper_metrics_effective_energy_eps = float(getattr(self.all_args, "auto_paper_metrics_effective_energy_eps", 1e-6))
        self.auto_paper_metrics_fidelity_metrics = [
            str(x) for x in getattr(self.all_args, "auto_paper_metrics_fidelity_metrics", ["advancement_factor", "cluster_count", "largest_cluster_size"])
        ]
        self.auto_paper_metrics_time_grid_points = int(getattr(self.all_args, "auto_paper_metrics_time_grid_points", 200))
        self.auto_paper_metrics_output_prefix = str(getattr(self.all_args, "auto_paper_metrics_output_prefix", "paper_metrics"))
        self.enable_auto_main_table1 = bool(getattr(self.all_args, "enable_auto_main_table1", False))
        self.auto_main_table1_script = str(getattr(self.all_args, "auto_main_table1_script", "scripts/build_main_table.py"))
        self.auto_main_table1_output_prefix = str(getattr(self.all_args, "auto_main_table1_output_prefix", "main_table1"))
        self.auto_main_table1_target_drop = float(getattr(self.all_args, "auto_main_table1_target_drop", -1.0))

        self.best_compare_metrics = {
            "final_energy_drop_diff": {"value": -float("inf"), "episode": -1},
            "auc_energy_drop_diff": {"value": -float("inf"), "episode": -1},
            "af_fixed_budget": {"value": -float("inf"), "episode": -1},
        }

    def plot_reward_energy_curve(self, episode):
        if not self.enable_episode_plots:
            return
        if self.plot_interval <= 0:
            return
        if episode % self.plot_interval != 0:
            return

        os.makedirs("plots", exist_ok=True)
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        step_axis = np.arange(len(self.reward_history))
        stride = max(1, len(self.reward_history) // 500) if len(self.reward_history) > 0 else 1
        window = min(200, max(1, len(self.reward_history) // 10)) if len(self.reward_history) > 0 else 1

        if len(self.reward_history) > 0:
            ax[0].plot(step_axis[::stride], np.asarray(self.reward_history)[::stride], alpha=0.2, color="gray")
            if len(self.reward_history) >= window:
                ax[0].plot(step_axis[window - 1 :], moving_average(self.reward_history, window), linewidth=2, color="blue")
        ax[0].set_ylabel("Reward")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title(f"Episode {episode} Metrics")

        if len(self.energy_change_history) > 0:
            ax[1].plot(step_axis[::stride], np.asarray(self.energy_change_history)[::stride], alpha=0.3, color="green")
            if len(self.energy_change_history) >= window:
                ax[1].plot(step_axis[window - 1 :], moving_average(self.energy_change_history, window), linewidth=2, color="darkgreen")
        ax[1].set_ylabel("Delta E (eV)")
        ax[1].grid(True, alpha=0.3)

        if len(self.abs_energy_history) > 0:
            ax[2].plot(step_axis, self.abs_energy_history, color="red", linewidth=1.0)
            ax[2].axhline(y=self.abs_energy_history[0], color="black", linestyle="--", alpha=0.5)
        ax[2].set_ylabel("Total Energy (eV)")
        ax[2].set_xlabel("Step")
        ax[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/episode_{episode}_summary.png", dpi=150)
        plt.close()

    def warmup(self):
        self.buffer.reset_rollout()
        obs, share_obs = self.envs.reset()
        self.buffer.obs[0] = obs
        self.buffer.share_obs[0] = np.asarray(share_obs[0], dtype=np.float32)
        self.buffer.positions[0] = np.asarray(self.envs.env.vacancy_pos, dtype=np.int32)
        self.buffer.base_rates[0] = np.asarray(self.envs.get_base_rates_np(copy=False), dtype=np.float32)
        self.buffer.event_features[0] = np.asarray(self.envs.get_event_features_np(copy=False), dtype=np.float32)
        sparse_tokens, sparse_masks = self.envs.get_sparse_defect_context_np(copy=False)
        self.buffer.sparse_defect_tokens[0] = np.asarray(sparse_tokens, dtype=np.float32)
        self.buffer.sparse_defect_masks[0] = np.asarray(sparse_masks, dtype=np.float32)
        self.buffer.masks[0] = np.ones((1, 1), dtype=np.float32)
        self.buffer.bad_masks[0] = np.ones((1, 1), dtype=np.float32)
        self.buffer.active_masks[0] = np.ones((1, 1), dtype=np.float32)

    @torch.no_grad()
    def collect(self, step):
        obs_batch = self.buffer.obs[step]
        share_obs_batch = self.buffer.share_obs[step]
        base_rates_batch = self.buffer.base_rates[step]
        event_features_batch = self.buffer.event_features[step]
        sparse_defect_tokens_batch = self.buffer.sparse_defect_tokens[step]
        sparse_defect_masks_batch = self.buffer.sparse_defect_masks[step]
        rnn_states_batch = self.buffer.rnn_states[step]
        rnn_states_critic_batch = self.buffer.rnn_states_critic[step]
        masks_batch = self.buffer.masks[step]
        values, actions, action_log_probs, rnn_states, rnn_states_critic, hazard_summary = self.trainer.policy.get_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            masks_batch,
            base_rates_batch,
            event_features_batch,
            sparse_defect_tokens=sparse_defect_tokens_batch,
            sparse_defect_masks=sparse_defect_masks_batch,
            deterministic=False,
        )
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, hazard_summary

    def insert(self, data):
        (
            obs,
            share_obs,
            positions,
            next_base_rates,
            next_event_features,
            next_sparse_defect_tokens,
            next_sparse_defect_masks,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            kmc_delta_t,
        ) = data

        done_flag = bool(np.asarray(dones).reshape(-1)[0])
        masks = np.zeros((1, 1), dtype=np.float32) if done_flag else np.ones((1, 1), dtype=np.float32)

        def _safe_t2n(x):
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            positions=positions,
            base_rates=next_base_rates,
            event_features=next_event_features,
            sparse_defect_tokens=next_sparse_defect_tokens,
            sparse_defect_masks=next_sparse_defect_masks,
            rnn_states=_safe_t2n(rnn_states),
            rnn_states_critic=_safe_t2n(rnn_states_critic),
            actions=_safe_t2n(actions),
            action_log_probs=_safe_t2n(action_log_probs),
            value_preds=_safe_t2n(values),
            rewards=rewards,
            kmc_delta_t=kmc_delta_t,
            masks=masks,
        )

    def _empty_episode_metrics(self):
        return {
            "valid_steps": 0,
            "reward_sum": 0.0,
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "energy_drop": 0.0,
            "mean_delta_e": 0.0,
            "std_delta_e": 0.0,
            "kmc_time_final": 0.0,
            "kmc_delta_t_mean": 0.0,
            "kmc_delta_t_sum": 0.0,
            "selected_rate_mean": 0.0,
            "selected_rate_max": 0.0,
            "total_rate_mean": 0.0,
            "pred_total_hazard_mean": 0.0,
            "pred_expected_dt_mean": 0.0,
            "pred_selected_hazard_mean": 0.0,
            "pred_log_total_hazard_mae": 0.0,
            "pred_log_expected_dt_mae": 0.0,
            "selected_over_total_mean": 0.0,
            "selected_over_total_std": 0.0,
            "legal_events_mean": 0.0,
            "legal_events_min": 0.0,
            "legal_events_max": 0.0,
            "done_flag": 0.0,
            "early_done": 0.0,
        }

    def summarize_episode_metrics(self, initial_energy, infos, valid_steps, done_flag):
        metrics = self._empty_episode_metrics()
        metrics["valid_steps"] = int(valid_steps)
        metrics["done_flag"] = float(done_flag)
        metrics["early_done"] = float(done_flag and valid_steps < self.episode_length)
        metrics["initial_energy"] = float(initial_energy)
        metrics["reward_sum"] = float(np.sum(self.buffer.rewards[:valid_steps])) if valid_steps > 0 else 0.0

        if len(self.abs_energy_history) > 0:
            final_energy = float(self.abs_energy_history[-1])
        else:
            final_energy = float(initial_energy)
        metrics["final_energy"] = final_energy
        metrics["energy_drop"] = float(initial_energy - final_energy)

        if len(self.energy_change_history) > 0:
            delta_e = np.asarray(self.energy_change_history, dtype=np.float64)
            metrics["mean_delta_e"] = float(np.mean(delta_e))
            metrics["std_delta_e"] = float(np.std(delta_e))

        if infos:
            kmc_delta_ts = np.asarray([info.get("kmc_delta_t", 0.0) for info in infos], dtype=np.float64)
            selected_rates = np.asarray([info.get("selected_rate", 0.0) for info in infos], dtype=np.float64)
            total_rates = np.asarray([info.get("total_rate", 0.0) for info in infos], dtype=np.float64)
            pred_total_hazards = np.asarray([info.get("pred_total_hazard", 0.0) for info in infos], dtype=np.float64)
            pred_expected_dts = np.asarray([info.get("pred_expected_dt", 0.0) for info in infos], dtype=np.float64)
            pred_selected_hazards = np.asarray([info.get("pred_selected_hazard", 0.0) for info in infos], dtype=np.float64)
            legal_counts = []
            for info in infos:
                legal_count = info.get("legal_event_count")
                if legal_count is None:
                    base_rates = np.asarray(info.get("base_rates", []), dtype=np.float64)
                    legal_counts.append(int(np.sum(base_rates > 0.0)))
                else:
                    legal_counts.append(int(legal_count))
            legal_counts = np.asarray(legal_counts, dtype=np.float64) if legal_counts else np.zeros((0,), dtype=np.float64)

            ratio = np.divide(selected_rates, np.maximum(total_rates, 1e-12))
            metrics["kmc_time_final"] = float(infos[-1].get("kmc_time", 0.0))
            metrics["kmc_delta_t_mean"] = float(np.mean(kmc_delta_ts)) if kmc_delta_ts.size > 0 else 0.0
            metrics["kmc_delta_t_sum"] = float(np.sum(kmc_delta_ts)) if kmc_delta_ts.size > 0 else 0.0
            metrics["selected_rate_mean"] = float(np.mean(selected_rates)) if selected_rates.size > 0 else 0.0
            metrics["selected_rate_max"] = float(np.max(selected_rates)) if selected_rates.size > 0 else 0.0
            metrics["total_rate_mean"] = float(np.mean(total_rates)) if total_rates.size > 0 else 0.0
            metrics["pred_total_hazard_mean"] = float(np.mean(pred_total_hazards)) if pred_total_hazards.size > 0 else 0.0
            metrics["pred_expected_dt_mean"] = float(np.mean(pred_expected_dts)) if pred_expected_dts.size > 0 else 0.0
            metrics["pred_selected_hazard_mean"] = float(np.mean(pred_selected_hazards)) if pred_selected_hazards.size > 0 else 0.0
            metrics["selected_over_total_mean"] = float(np.mean(ratio)) if ratio.size > 0 else 0.0
            metrics["selected_over_total_std"] = float(np.std(ratio)) if ratio.size > 0 else 0.0
            metrics["legal_events_mean"] = float(np.mean(legal_counts)) if legal_counts.size > 0 else 0.0
            metrics["legal_events_min"] = float(np.min(legal_counts)) if legal_counts.size > 0 else 0.0
            metrics["legal_events_max"] = float(np.max(legal_counts)) if legal_counts.size > 0 else 0.0

            valid_rate_mask = (total_rates > 0.0) & (pred_total_hazards > 0.0)
            if np.any(valid_rate_mask):
                metrics["pred_log_total_hazard_mae"] = float(
                    np.mean(
                        np.abs(
                            np.log(pred_total_hazards[valid_rate_mask])
                            - np.log(total_rates[valid_rate_mask])
                        )
                    )
                )
            valid_dt_mask = valid_rate_mask & (pred_expected_dts > 0.0)
            if np.any(valid_dt_mask):
                target_expected_dt = 1.0 / np.maximum(total_rates[valid_dt_mask], 1e-12)
                metrics["pred_log_expected_dt_mae"] = float(
                    np.mean(
                        np.abs(
                            np.log(pred_expected_dts[valid_dt_mask])
                            - np.log(target_expected_dt)
                        )
                    )
                )
        return metrics

    def _resolve_snapshot_output_root(self):
        if self.snapshot_panel_output_root:
            return Path(self.snapshot_panel_output_root)
        if hasattr(self, "run_dir"):
            return Path(self.run_dir) / "snapshot_panels"
        return Path(self.save_dir).parent / "snapshot_panels"

    def _resolve_snapshot_steps(self) -> List[int]:
        configured = self.snapshot_panel_steps
        if configured:
            steps = sorted({max(0, int(x)) for x in configured})
        else:
            if self.episode_length <= 0:
                steps = [0]
            else:
                steps = sorted({int(x) for x in np.linspace(0, self.episode_length, num=4)})
        return steps if steps else [0]

    def _snapshot_panel_due(self, episode: int, episodes: int) -> bool:
        if not self.enable_snapshot_panels:
            return False
        return ((episode + 1) % self.snapshot_panel_interval == 0) or (episode == episodes - 1)

    def _init_snapshot_capture(self, targets: List[int]):
        capture = {
            "targets": list(targets),
            "next_idx": 0,
            "lattices": [],
            "times": [],
            "realized_steps": [],
            "filled_from_final": [],
        }
        self._record_snapshot_frames(capture, current_step=0)
        return capture

    def _record_snapshot_frames(self, capture, current_step: int):
        env = self.envs.env
        while capture["next_idx"] < len(capture["targets"]) and current_step >= capture["targets"][capture["next_idx"]]:
            capture["lattices"].append(np.asarray(env.lattice, dtype=np.int32).copy())
            capture["times"].append(float(env.time))
            capture["realized_steps"].append(int(current_step))
            capture["filled_from_final"].append(False)
            capture["next_idx"] += 1

    def _finalize_snapshot_capture(self, capture, final_step: int):
        env = self.envs.env
        while capture["next_idx"] < len(capture["targets"]):
            capture["lattices"].append(np.asarray(env.lattice, dtype=np.int32).copy())
            capture["times"].append(float(env.time))
            capture["realized_steps"].append(int(final_step))
            capture["filled_from_final"].append(True)
            capture["next_idx"] += 1

    def _snapshot_titles_from_capture(self, capture) -> List[str]:
        titles = []
        for target, realized, filled in zip(
            capture["targets"],
            capture["realized_steps"],
            capture["filled_from_final"],
        ):
            if filled and realized != target:
                titles.append(f"step={target}\nfinal@{realized}")
            else:
                titles.append(f"step={target}")
        return titles

    def save_snapshot_panel(self, episode: int, total_num_steps: int, capture) -> str:
        output_root = self._resolve_snapshot_output_root()
        output_root.mkdir(parents=True, exist_ok=True)
        save_path = output_root / f"episode_{episode:05d}_step_{int(total_num_steps):09d}_cu_panel.png"
        titles = self._snapshot_titles_from_capture(capture)
        suptitle = f"Episode {episode} Cu clustering evolution"
        self.envs.env.visualize_snapshots(
            lattices=capture["lattices"],
            titles=titles,
            save_path=str(save_path),
            show_vacancy=self.snapshot_panel_show_vacancy,
            suptitle=suptitle,
        )
        return str(save_path)

    def _resolve_auto_compare_output_root(self):
        if self.auto_compare_output_root:
            return Path(self.auto_compare_output_root)
        if hasattr(self, "run_dir"):
            return Path(self.run_dir) / "auto_compare"
        return Path(self.save_dir).parent / "auto_compare"

    def _resolve_helper_script(self, script_path: str, label: str):
        candidate = Path(script_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        search_roots = [Path(os.getcwd())]
        try:
            search_roots.extend([Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]])
        except Exception:
            pass

        for root in search_roots:
            probe = (root / script_path).resolve()
            if probe.exists():
                return probe
        raise FileNotFoundError(f"{label} script not found: {script_path}")

    def _resolve_auto_compare_script(self):
        return self._resolve_helper_script(self.auto_compare_script, "Auto compare")

    def _resolve_auto_paper_metrics_script(self):
        return self._resolve_helper_script(self.auto_paper_metrics_script, "Auto paper metrics")

    def _resolve_auto_main_table1_script(self):
        return self._resolve_helper_script(self.auto_main_table1_script, "Auto main Table 1")

    def _build_auto_compare_command(self, compare_script: Path, output_dir: Path, current_scale: float, current_rate_prior_scale: float, current_energy_prior_scale: float, model_dir=None):
        cmd = [
            self.auto_compare_python_bin,
            str(compare_script),
            "--model_dir",
            str(model_dir if model_dir is not None else self.save_dir),
            "--compare_workers",
            str(self.auto_compare_workers),
            "--compare_output_dir",
            str(output_dir),
            "--bench_step",
            str(self.auto_compare_bench_step if self.auto_compare_bench_step > 0 else self.episode_length),
            "--episode_length",
            str(self.auto_compare_bench_step if self.auto_compare_bench_step > 0 else self.episode_length),
            "--energy_series_interval_steps",
            str(self.auto_compare_energy_series_interval_steps),
            "--compare_seeds",
            *[str(x) for x in self.auto_compare_seeds],
            "--compare_modes",
            *self.auto_compare_modes,
            "--plot_band",
            self.auto_compare_plot_band,
            "--bootstrap_samples",
            str(self.auto_compare_bootstrap_samples),
            "--target_energy_drops",
            *[str(x) for x in self.auto_compare_target_energy_drops],
            "--rate_energy_beta",
            str(self.auto_compare_rate_energy_beta),
            "--temperature",
            str(self.all_args.temperature),
            "--lattice_size",
            *[str(x) for x in self.all_args.lattice_size],
            "--lattice_cu_nums",
            str(self.all_args.lattice_cu_nums),
            "--lattice_v_nums",
            str(self.all_args.lattice_v_nums),
            "--hidden_size",
            str(self.all_args.hidden_size),
            "--history_len",
            str(getattr(self.all_args, "history_len", 5)),
            "--dense_local_scale",
            str(getattr(self.all_args, "dense_local_scale", 1.0)),
            "--gnn_attention_heads",
            str(getattr(self.all_args, "gnn_attention_heads", 4)),
            "--gnn_attention_dropout",
            str(getattr(self.all_args, "gnn_attention_dropout", 0.1)),
            "--gnn_layers",
            str(getattr(self.all_args, "gnn_layers", 3)),
            "--neighbor_radius",
            str(getattr(self.all_args, "neighbor_radius", 4.0)),
            "--max_num_neighbors",
            str(getattr(self.all_args, "max_num_neighbors", 64)),
            "--sparse_defect_attention_heads",
            str(getattr(self.all_args, "sparse_defect_attention_heads", 4)),
            "--sparse_defect_transformer_layers",
            str(getattr(self.all_args, "sparse_defect_transformer_layers", 1)),
            "--sparse_defect_dropout",
            str(getattr(self.all_args, "sparse_defect_dropout", 0.1)),
            "--sparse_defect_offset_scale",
            str(getattr(self.all_args, "sparse_defect_offset_scale", 8.0)),
            "--sparse_defect_shell_cutoff",
            str(getattr(self.all_args, "sparse_defect_shell_cutoff", 16)),
            "--sparse_defect_max_tokens",
            str(getattr(self.all_args, "sparse_defect_max_tokens", 24)),
            "--direction_attention_heads",
            str(getattr(self.all_args, "direction_attention_heads", 4)),
            "--direction_attention_dropout",
            str(getattr(self.all_args, "direction_attention_dropout", 0.05)),
            "--policy_logit_scale",
            str(current_scale),
            "--rate_prior_scale",
            str(current_rate_prior_scale),
            "--energy_prior_scale",
            str(current_energy_prior_scale),
            "--energy_prior_downhill_bonus",
            str(getattr(self.all_args, "energy_prior_downhill_bonus", 0.20)),
            "--energy_prior_cu_bonus",
            str(getattr(self.all_args, "energy_prior_cu_bonus", 0.10)),
            "--energy_prior_fe_penalty",
            str(getattr(self.all_args, "energy_prior_fe_penalty", 0.04)),
            "--energy_prior_clip",
            str(getattr(self.all_args, "energy_prior_clip", 3.0)),
        ]

        if bool(getattr(self.all_args, "use_direction_aware_head", True)):
            cmd.append("--use_direction_aware_head")
        if not bool(getattr(self.all_args, "use_sparse_context", True)):
            cmd.append("--disable_sparse_context")
        if bool(getattr(self.all_args, "reserve_other_vacancy_tokens", False)):
            cmd.append("--reserve_other_vacancy_tokens")
        if not bool(getattr(self.all_args, "use_action_history", True)):
            cmd.append("--disable_action_history")
        if not bool(getattr(self.all_args, "use_recurrent_memory", True)):
            cmd.append("--disable_recurrent_memory")

        if self.device.type == "cuda":
            cmd.extend(["--cuda", "--device", str(self.device)])
        else:
            cmd.extend(["--no_cuda", "--device", "cpu"])
        return cmd

    def _build_auto_paper_metrics_command(self, paper_metrics_script: Path, compare_output_dir: Path):
        cmd = [
            self.auto_compare_python_bin,
            str(paper_metrics_script),
            "--compare_dir",
            str(compare_output_dir),
            "--reference_mode",
            self.auto_paper_metrics_reference_mode,
            "--main_mode",
            self.auto_paper_metrics_main_mode,
            "--method_label",
            self.auto_paper_metrics_method_label,
            "--effective_energy_eps",
            str(self.auto_paper_metrics_effective_energy_eps),
            "--time_grid_points",
            str(self.auto_paper_metrics_time_grid_points),
            "--output_prefix",
            self.auto_paper_metrics_output_prefix,
        ]
        if self.auto_paper_metrics_fidelity_metrics:
            cmd.extend(["--fidelity_metrics", *self.auto_paper_metrics_fidelity_metrics])
        return cmd

    def _build_auto_main_table1_command(self, table1_script: Path, compare_output_dir: Path):
        cmd = [
            self.auto_compare_python_bin,
            str(table1_script),
            "--compare_dir",
            str(compare_output_dir),
            "--paper_prefix",
            self.auto_paper_metrics_output_prefix,
            "--reference_mode",
            self.auto_paper_metrics_reference_mode,
            "--output_prefix",
            self.auto_main_table1_output_prefix,
        ]
        if self.auto_main_table1_target_drop > 0.0:
            cmd.extend(["--target_energy_drop", str(self.auto_main_table1_target_drop)])
        return cmd

    def _write_best_compare_metadata(self, target_dir: Path, metric_name: str, metric_value: float, episode: int, total_num_steps: int, compare_output_dir: Path, current_scale: float, current_rate_prior_scale: float, current_energy_prior_scale: float):
        payload = {
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "episode": int(episode),
            "total_num_steps": int(total_num_steps),
            "compare_output_dir": str(compare_output_dir),
            "policy_logit_scale": float(current_scale),
            "rate_prior_scale": float(current_rate_prior_scale),
            "energy_prior_scale": float(current_energy_prior_scale),
        }
        with (target_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _maybe_update_best_compare_model(self, metric_name: str, metric_value: float, episode: int, total_num_steps: int, compare_output_dir: Path, current_scale: float, current_rate_prior_scale: float, current_energy_prior_scale: float):
        if not np.isfinite(metric_value):
            return None
        best_record = self.best_compare_metrics[metric_name]
        if metric_value <= best_record["value"]:
            return None

        best_record["value"] = float(metric_value)
        best_record["episode"] = int(episode)
        best_root = self._resolve_auto_compare_output_root() / "best_models" / metric_name
        best_root.mkdir(parents=True, exist_ok=True)
        self.save(best_root)
        self._write_best_compare_metadata(best_root, metric_name, metric_value, episode, total_num_steps, compare_output_dir, current_scale, current_rate_prior_scale, current_energy_prior_scale)
        return str(best_root)

    def run_auto_compare(self, episode: int, total_num_steps: int, current_scale: float, current_rate_prior_scale: float, current_energy_prior_scale: float):
        compare_script = self._resolve_auto_compare_script()
        output_dir = self._resolve_auto_compare_output_root() / f"episode_{episode:05d}_step_{int(total_num_steps):09d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir = Path(self.save_dir) / "candidates" / f"round_{episode + 1:05d}"
        self.save(candidate_dir)
        cmd = self._build_auto_compare_command(compare_script, output_dir, current_scale, current_rate_prior_scale, current_energy_prior_scale, model_dir=candidate_dir)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MPLBACKEND", "Agg")
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (output_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (output_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")

        compare_payload = {
            "auto_compare/return_code": float(result.returncode),
        }
        if result.returncode != 0:
            print(f"[AutoCompare] failed at episode {episode} with return code {result.returncode}. See {output_dir}")
            if self.auto_compare_require_success:
                raise RuntimeError(f"Scheduled comparison failed: {output_dir}")
            return compare_payload

        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            print(f"[AutoCompare] summary.json not found at {summary_path}")
            if self.auto_compare_require_success:
                raise RuntimeError(f"Scheduled comparison missing summary: {output_dir}")
            return compare_payload

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        af_record = summary.get("fixed_budget_af", {}).get(self.auto_paper_metrics_main_mode, {})
        af_metric = float(af_record.get("speedup", np.nan))
        af_hits, af_total = int(af_record.get("n_hit", 0)), int(af_record.get("n_total", 0))
        eligible = af_total > 0 and af_hits == af_total and np.isfinite(af_metric)
        selection_file = self._resolve_auto_compare_output_root() / "af_selection_history.json"
        selection_history = json.loads(selection_file.read_text()) if selection_file.exists() else []
        selection_history.append({"round": episode + 1, "train_steps": total_num_steps,
                                  "af": af_metric if np.isfinite(af_metric) else None,
                                  "target_hits": af_hits, "seeds": af_total, "eligible": bool(eligible),
                                  "checkpoint": str(candidate_dir), "comparison": str(output_dir)})
        selection_file.write_text(json.dumps(selection_history, indent=2), encoding="utf-8")
        best_af_dir = self._maybe_update_best_compare_model(
            "af_fixed_budget", af_metric if eligible else np.nan, episode, total_num_steps,
            output_dir, current_scale, current_rate_prior_scale, current_energy_prior_scale,
        )
        compare_payload["auto_compare/af"] = af_metric
        compare_payload["auto_compare/af_target_hits"] = float(af_hits)
        if best_af_dir is not None:
            print(f"[AutoCompare] new best AF={af_metric:.6g} ({af_hits}/{af_total}) at round {episode + 1}. Saved to {best_af_dir}")
        paired = summary.get("paired_statistics", {})
        final_metric = float(paired.get("final_energy_drop_diff", {}).get("mean", np.nan))
        auc_metric = float(paired.get("auc_energy_drop_diff", {}).get("mean", np.nan))

        compare_payload.update({
            "auto_compare/final_energy_drop_diff_mean": final_metric,
            "auto_compare/auc_energy_drop_diff_mean": auc_metric,
        })

        if self.enable_auto_paper_metrics:
            try:
                paper_metrics_script = self._resolve_auto_paper_metrics_script()
                paper_cmd = self._build_auto_paper_metrics_command(paper_metrics_script, output_dir)
                paper_result = subprocess.run(paper_cmd, capture_output=True, text=True, env=env)
                (output_dir / "paper_metrics_stdout.log").write_text(paper_result.stdout or "", encoding="utf-8")
                (output_dir / "paper_metrics_stderr.log").write_text(paper_result.stderr or "", encoding="utf-8")
                compare_payload["auto_compare/paper_metrics_return_code"] = float(paper_result.returncode)
                if paper_result.returncode != 0:
                    if self.auto_compare_require_success:
                        raise RuntimeError(f"Scheduled paper metrics failed: {output_dir}")
                    print(
                        f"[AutoPaperMetrics] failed at episode {episode} with return code {paper_result.returncode}. See {output_dir}"
                    )
                else:
                    paper_summary_path = output_dir / f"{self.auto_paper_metrics_output_prefix}_summary.json"
                    if paper_summary_path.exists():
                        paper_summary = json.loads(paper_summary_path.read_text(encoding="utf-8"))
                        main_rows = paper_summary.get("main_row", [])
                        if main_rows:
                            main_row = main_rows[0]
                            compare_payload["auto_compare/paper_eds_mean"] = float(
                                main_row.get("energy_descent_per_step_mean", np.nan)
                            )
                            compare_payload["auto_compare/paper_etr_mean"] = float(
                                main_row.get("effective_transition_ratio_mean", np.nan)
                            )
                            compare_payload["auto_compare/paper_dynerr_mean"] = float(
                                main_row.get("dynerr_composite_mean", np.nan)
                            )
                    print(
                        f"[AutoPaperMetrics] episode={episode}, output={output_dir / (self.auto_paper_metrics_output_prefix + '_summary.json')}"
                    )
                    if self.enable_auto_main_table1:
                        table1_script = self._resolve_auto_main_table1_script()
                        table1_cmd = self._build_auto_main_table1_command(table1_script, output_dir)
                        table1_result = subprocess.run(table1_cmd, capture_output=True, text=True, env=env)
                        (output_dir / "main_table1_stdout.log").write_text(table1_result.stdout or "", encoding="utf-8")
                        (output_dir / "main_table1_stderr.log").write_text(table1_result.stderr or "", encoding="utf-8")
                        compare_payload["auto_compare/main_table1_return_code"] = float(table1_result.returncode)
                        if table1_result.returncode != 0:
                            if self.auto_compare_require_success:
                                raise RuntimeError(f"Scheduled metric table failed: {output_dir}")
                            print(
                                f"[AutoMainTable1] failed at episode {episode} with return code {table1_result.returncode}. See {output_dir}"
                            )
                        else:
                            print(
                                f"[AutoMainTable1] episode={episode}, output={output_dir / (self.auto_main_table1_output_prefix + '.display.csv')}"
                            )
            except Exception as exc:
                if self.auto_compare_require_success:
                    raise
                compare_payload["auto_compare/paper_metrics_return_code"] = -1.0
                print(f"[AutoPaperMetrics] failed to run at episode {episode}: {exc}")

        best_final_dir = self._maybe_update_best_compare_model(
            "final_energy_drop_diff", final_metric, episode, total_num_steps, output_dir, current_scale, current_rate_prior_scale, current_energy_prior_scale
        )
        best_auc_dir = self._maybe_update_best_compare_model(
            "auc_energy_drop_diff", auc_metric, episode, total_num_steps, output_dir, current_scale, current_rate_prior_scale, current_energy_prior_scale
        )

        compare_payload["auto_compare/best_final_energy_drop_diff_mean_so_far"] = float(
            self.best_compare_metrics["final_energy_drop_diff"]["value"]
        )
        compare_payload["auto_compare/best_auc_energy_drop_diff_mean_so_far"] = float(
            self.best_compare_metrics["auc_energy_drop_diff"]["value"]
        )

        if best_final_dir is not None:
            print(
                f"[AutoCompare] new best final_energy_drop_diff={final_metric:.4f} at episode {episode}. Saved to {best_final_dir}"
            )
        if best_auc_dir is not None:
            print(
                f"[AutoCompare] new best auc_energy_drop_diff={auc_metric:.4f} at episode {episode}. Saved to {best_auc_dir}"
            )

        print(
            f"[AutoCompare] episode={episode}, final_diff={final_metric:.4f}, auc_diff={auc_metric:.4f}, output={output_dir}"
        )
        return compare_payload

    def run(self):
        overall_start = time.time()
        cumulative_collect_time = 0.0
        cumulative_train_time = 0.0
        total_completed_steps = 0
        episodes = int(self.num_env_steps) // self.episode_length
        snapshot_targets = self._resolve_snapshot_steps() if self.enable_snapshot_panels else []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            current_policy_logit_scale = float(self.trainer.policy.update_policy_logit_scale(episode, episodes))
            current_rate_prior_scale = float(self.trainer.policy.rate_prior_scale)
            current_energy_prior_scale = float(getattr(self.trainer.policy, "energy_prior_scale", 0.0))

            self.reward_history = []
            self.energy_change_history = []
            self.abs_energy_history = []
            self.warmup()
            initial_energy = self.envs.env.calculate_system_energy()
            episode_infos = []
            done_flag = False
            snapshot_capture = self._init_snapshot_capture(snapshot_targets) if self._snapshot_panel_due(episode, episodes) else None

            collect_start = time.time()
            self.trainer.prep_rollout()
            step_iter = range(self.episode_length)
            if self.show_collect_progress:
                step_iter = tqdm(step_iter, desc=f"Episode {episode} Collecting")
            for step in step_iter:
                values, actions, action_log_probs, rnn_states, rnn_states_critic, hazard_summary = self.collect(step)
                obs, share_obs, positions, rewards, dones, info = self.envs.step(actions, episode)
                next_base_rates = np.asarray(
                    info.get("next_base_rates", self.envs.get_base_rates_np(copy=False)),
                    dtype=np.float32,
                )
                next_event_features = np.asarray(
                    info.get("next_event_features", self.envs.get_event_features_np(copy=False)),
                    dtype=np.float32,
                )
                next_sparse_defect_tokens = info.get("next_sparse_defect_tokens")
                next_sparse_defect_masks = info.get("next_sparse_defect_masks")
                if next_sparse_defect_tokens is None or next_sparse_defect_masks is None:
                    next_sparse_defect_tokens, next_sparse_defect_masks = self.envs.get_sparse_defect_context_np(copy=False)
                next_sparse_defect_tokens = np.asarray(next_sparse_defect_tokens, dtype=np.float32)
                next_sparse_defect_masks = np.asarray(next_sparse_defect_masks, dtype=np.float32)

                self.reward_history.append(float(np.asarray(rewards).reshape(-1)[0]))
                self.energy_change_history.append(float(info.get("global_energy_change", 0.0)))
                self.abs_energy_history.append(float(info.get("total_energy", initial_energy)))
                episode_infos.append(
                    {
                        "kmc_delta_t": float(info.get("kmc_delta_t", 0.0)),
                        "kmc_time": float(info.get("kmc_time", 0.0)),
                        "selected_rate": float(info.get("selected_rate", 0.0)),
                        "total_rate": float(info.get("total_rate", 0.0)),
                        "pred_total_hazard": float(np.asarray(hazard_summary["pred_total_hazard"]).reshape(-1)[0]),
                        "pred_expected_dt": float(np.asarray(hazard_summary["pred_expected_dt"]).reshape(-1)[0]),
                        "pred_selected_hazard": float(np.asarray(hazard_summary["pred_selected_hazard"]).reshape(-1)[0]),
                        "legal_event_count": int(info.get("legal_event_count", 0)),
                    }
                )

                data = (
                    obs,
                    share_obs,
                    positions,
                    next_base_rates,
                    next_event_features,
                    next_sparse_defect_tokens,
                    next_sparse_defect_masks,
                    rewards,
                    dones,
                    info,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    np.asarray([[info.get("kmc_delta_t", 0.0)]], dtype=np.float32),
                )
                self.insert(data)

                if snapshot_capture is not None:
                    self._record_snapshot_frames(snapshot_capture, current_step=step + 1)

                done_flag = bool(np.asarray(dones).reshape(-1)[0])
                if done_flag:
                    break
            collect_time = time.time() - collect_start
            cumulative_collect_time += collect_time

            self.buffer.finalize_trajectory()
            valid_steps = int(self.buffer.valid_length)
            total_completed_steps += valid_steps
            self.plot_reward_energy_curve(episode)

            train_start = time.time()
            self.compute()
            train_infos = self.train()
            train_time = time.time() - train_start
            cumulative_train_time += train_time

            total_num_steps = total_completed_steps
            episode_metrics = self.summarize_episode_metrics(initial_energy, episode_infos, valid_steps, done_flag)

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            compare_payload = {}
            auto_compare_due = self.enable_auto_compare and (((episode + 1) % self.auto_compare_interval == 0) or (episode == episodes - 1))
            if auto_compare_due:
                self.save()
                compare_payload = self.run_auto_compare(episode, total_num_steps, current_policy_logit_scale, current_rate_prior_scale, current_energy_prior_scale)
                if compare_payload:
                    if self.use_wandb:
                        wandb.log(compare_payload, step=total_num_steps)
                    else:
                        for key, value in compare_payload.items():
                            self.writter.add_scalar(key, value, total_num_steps)

            snapshot_path = ""
            if snapshot_capture is not None:
                self._finalize_snapshot_capture(snapshot_capture, final_step=valid_steps)
                snapshot_path = self.save_snapshot_panel(episode, total_num_steps, snapshot_capture)
                print(f"[Snapshots] saved Cu panel to {snapshot_path}")

            if episode % self.log_interval == 0:
                wall_time = time.time() - overall_start
                collect_sps = valid_steps / max(collect_time, 1e-6)
                overall_sps = total_completed_steps / max(wall_time, 1e-6)
                print("\n" + "=" * 80)
                print(f" Episode      : {episode:<5d} / {episodes:<5d} | Valid Steps: {valid_steps}")
                print(f" Collect Time : {collect_time:>8.2f}s | Train Time: {train_time:>8.2f}s | Total: {wall_time:>8.2f}s")
                print(f" Collect SPS  : {collect_sps:>8.2f}  | Overall SPS: {overall_sps:>8.2f}")
                print("-" * 80)
                print(f" Reward Sum   : {episode_metrics['reward_sum']:<10.4f}")
                print(f" Energy Drop  : {episode_metrics['energy_drop']:<10.4f} eV")
                print(f" Final Energy : {episode_metrics['final_energy']:<10.4f} eV")
                print(f" KMC Time     : {episode_metrics['kmc_time_final']:<10.4e} | Mean dt: {episode_metrics['kmc_delta_t_mean']:<10.4e}")
                print(f" Pred Hazard  : {episode_metrics['pred_total_hazard_mean']:<10.4e} | Pred dt: {episode_metrics['pred_expected_dt_mean']:<10.4e}")
                print(f" Time Calib   : log-rate MAE={episode_metrics['pred_log_total_hazard_mae']:<10.4f} | log-dt MAE={episode_metrics['pred_log_expected_dt_mae']:<10.4f}")
                print(f" Sel/Total    : {episode_metrics['selected_over_total_mean']:<10.4e} | Legal Events: {episode_metrics['legal_events_mean']:<10.2f}")
                print("-" * 80)
                print(f" Policy Scale : {current_policy_logit_scale:.4f}")
                print(f" Rate Prior   : {current_rate_prior_scale:.4f}")
                print(f" Energy Prior : {current_energy_prior_scale:.4f}")
                print(f" Entropy      : {train_infos['dist_entropy']:.4f}  | Value Loss: {train_infos['value_loss']:.4f}")
                print(f" Aux PhysLoss : {train_infos['aux_phys_loss']:.4f}")
                print(f" Aux EventTime: {train_infos['aux_event_time_loss']:.4f} | Event KL: {train_infos['event_kl']:.4f} | Time NLLΔ: {train_infos['time_nll_residual']:.4f}")
                print(f" Policy Loss  : {train_infos['policy_loss']:.4f}  | Grad Norm : {train_infos['actor_grad_norm']:.4f}")
                if compare_payload:
                    print(
                        f" AutoCompare  : final_diff={compare_payload.get('auto_compare/final_energy_drop_diff_mean', float('nan')):.4f} | "
                        f"auc_diff={compare_payload.get('auto_compare/auc_energy_drop_diff_mean', float('nan')):.4f}"
                    )
                if snapshot_path:
                    print(f" Snapshot     : {snapshot_path}")
                print("=" * 80 + "\n")

                log_payload = {
                    "episode": episode,
                    "total_steps": total_num_steps,
                    "episode_valid_steps": valid_steps,
                    "episode_collect_time": collect_time,
                    "episode_train_time": train_time,
                    "episode_wall_time": collect_time + train_time,
                    "collect_steps_per_sec": collect_sps,
                    "overall_steps_per_sec": overall_sps,
                    "cumulative_collect_time": cumulative_collect_time,
                    "cumulative_train_time": cumulative_train_time,
                    "policy_logit_scale": current_policy_logit_scale,
                    "rate_prior_scale": current_rate_prior_scale,
                    "energy_prior_scale": current_energy_prior_scale,
                    "policy_entropy": train_infos["dist_entropy"],
                    "value_loss": train_infos["value_loss"],
                    "policy_loss": train_infos["policy_loss"],
                    "aux_phys_loss": train_infos["aux_phys_loss"],
                    "aux_event_time_loss": train_infos["aux_event_time_loss"],
                    "event_kl": train_infos["event_kl"],
                    "time_nll_residual": train_infos["time_nll_residual"],
                    "log_total_hazard_mae": train_infos["log_total_hazard_mae"],
                    "log_expected_dt_mae": train_infos["log_expected_dt_mae"],
                    "actor_grad_norm": train_infos["actor_grad_norm"],
                }
                if snapshot_path:
                    log_payload["snapshot_panel/episode_index"] = float(episode)
                for key, value in episode_metrics.items():
                    log_payload[f"episode/{key}"] = value
                log_payload.update(compare_payload)

                if self.use_wandb:
                    wandb.log(log_payload, step=total_num_steps)
                else:
                    for key, value in log_payload.items():
                        self.writter.add_scalar(key, value, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, episode)

        if self.use_wandb:
            wandb.finish()

    @torch.no_grad()
    def eval(self, total_num_steps, episode):
        eval_summaries = []
        eval_episodes = int(getattr(self.all_args, "eval_episodes", 1))
        print_eval_details = bool(getattr(self.all_args, "print_eval_details", False))

        for eval_ep in range(eval_episodes):
            eval_obs, eval_share_obs = self.eval_envs.reset()
            eval_rnn_states = np.zeros((self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, 1), dtype=np.float32)
            eval_base_rates = np.asarray(self.eval_envs.get_base_rates_np(copy=False), dtype=np.float32)
            eval_event_features = np.asarray(self.eval_envs.get_event_features_np(copy=False), dtype=np.float32)
            eval_sparse_tokens, eval_sparse_masks = self.eval_envs.get_sparse_defect_context_np(copy=False)
            initial_energy = float(self.eval_envs.env.calculate_system_energy())

            reward_history = []
            delta_e_history = []
            energy_history = []
            info_history = []
            done_flag = False

            self.trainer.prep_rollout()
            step_iter = range(self.episode_length)
            if self.show_eval_progress:
                step_iter = tqdm(step_iter, desc=f"Evaluating ep {eval_ep}")
            for _ in step_iter:
                eval_actions, eval_rnn_states, eval_hazard_summary = self.trainer.policy.act(
                    eval_obs,
                    eval_rnn_states,
                    eval_masks,
                    eval_base_rates,
                    eval_event_features,
                    sparse_defect_tokens=eval_sparse_tokens,
                    sparse_defect_masks=eval_sparse_masks,
                    deterministic=True,
                    return_hazard_summary=True,
                )
                eval_obs, eval_share_obs, _, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions, episode)

                reward_history.append(float(np.asarray(eval_rewards).reshape(-1)[0]))
                delta_e_history.append(float(eval_infos.get("global_energy_change", 0.0)))
                energy_history.append(float(eval_infos.get("total_energy", initial_energy)))
                info_history.append(
                    {
                        "kmc_delta_t": float(eval_infos.get("kmc_delta_t", 0.0)),
                        "kmc_time": float(eval_infos.get("kmc_time", 0.0)),
                        "selected_rate": float(eval_infos.get("selected_rate", 0.0)),
                        "total_rate": float(eval_infos.get("total_rate", 0.0)),
                        "pred_total_hazard": float(np.asarray(eval_hazard_summary["pred_total_hazard"]).reshape(-1)[0]),
                        "pred_expected_dt": float(np.asarray(eval_hazard_summary["pred_expected_dt"]).reshape(-1)[0]),
                        "pred_selected_hazard": float(np.asarray(eval_hazard_summary["pred_selected_hazard"]).reshape(-1)[0]),
                        "legal_event_count": int(eval_infos.get("legal_event_count", 0)),
                    }
                )

                done_flag = bool(np.asarray(eval_dones).reshape(-1)[0])
                eval_masks = np.zeros((1, 1), dtype=np.float32) if done_flag else np.ones((1, 1), dtype=np.float32)
                eval_base_rates = np.asarray(
                    eval_infos.get("next_base_rates", self.eval_envs.get_base_rates_np(copy=False)),
                    dtype=np.float32,
                )
                eval_event_features = np.asarray(
                    eval_infos.get("next_event_features", self.eval_envs.get_event_features_np(copy=False)),
                    dtype=np.float32,
                )
                eval_sparse_tokens = eval_infos.get("next_sparse_defect_tokens")
                eval_sparse_masks = eval_infos.get("next_sparse_defect_masks")
                if eval_sparse_tokens is None or eval_sparse_masks is None:
                    eval_sparse_tokens, eval_sparse_masks = self.eval_envs.get_sparse_defect_context_np(copy=False)
                eval_sparse_tokens = np.asarray(eval_sparse_tokens, dtype=np.float32)
                eval_sparse_masks = np.asarray(eval_sparse_masks, dtype=np.float32)
                if done_flag:
                    break

            valid_steps = len(reward_history)
            summary = self.summarize_episode_metrics(initial_energy, info_history, valid_steps, done_flag)
            summary["reward_sum"] = float(np.sum(reward_history)) if reward_history else 0.0
            if delta_e_history:
                summary["mean_delta_e"] = float(np.mean(delta_e_history))
                summary["std_delta_e"] = float(np.std(delta_e_history))
            eval_summaries.append(summary)

            if print_eval_details:
                print(
                    f"[Eval episode {eval_ep}] steps={valid_steps}, reward_sum={summary['reward_sum']:.4f}, "
                    f"energy_drop={summary['energy_drop']:.4f}, kmc_time={summary['kmc_time_final']:.4e}, "
                    f"sel/total={summary['selected_over_total_mean']:.4e}, legal_events={summary['legal_events_mean']:.2f}"
                )

        aggregate = {}
        if eval_summaries:
            keys = sorted(eval_summaries[0].keys())
            for key in keys:
                values = np.asarray([summary[key] for summary in eval_summaries], dtype=np.float64)
                aggregate[f"eval/{key}_mean"] = float(np.mean(values))
                aggregate[f"eval/{key}_std"] = float(np.std(values))

        if self.use_wandb:
            wandb.log(aggregate, step=total_num_steps)
        else:
            for key, value in aggregate.items():
                self.writter.add_scalar(key, value, total_num_steps)

        if aggregate:
            print(
                "[Eval summary] "
                f"reward_sum={aggregate.get('eval/reward_sum_mean', 0.0):.4f}±{aggregate.get('eval/reward_sum_std', 0.0):.4f}, "
                f"energy_drop={aggregate.get('eval/energy_drop_mean', 0.0):.4f}±{aggregate.get('eval/energy_drop_std', 0.0):.4f}, "
                f"kmc_time={aggregate.get('eval/kmc_time_final_mean', 0.0):.4e}, "
                f"log_rate_mae={aggregate.get('eval/pred_log_total_hazard_mae_mean', 0.0):.4f}, "
                f"sel/total={aggregate.get('eval/selected_over_total_mean_mean', 0.0):.4e}, "
                f"legal_events={aggregate.get('eval/legal_events_mean_mean', 0.0):.2f}"
            )

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_idx = int(self.buffer.valid_length)
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
