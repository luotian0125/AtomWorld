"""Evaluate a BCC policy and an independently calibrated residence checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .bcc_clusters import CuClusters

METHODS = ('classical', 'rate_greedy', 'energy_greedy', 'full',
           'snapshot_from_full')


def evaluate(args):
    import torch
    torch.set_num_threads(1)
    from .backends.bcc.config.parser import get_config
    from .backends.bcc.envs.kmc_env import KMCEnvWrap
    from .backends.bcc.ppo.policy import R_MAPPOPolicy
    from .backends.bcc.scripts.compare_methods import (
        apply_checkpoint_config, sample_rl_action, sample_heuristic_action, seed_everything)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    clock_path = None
    learned_clock = args.method in ('full', 'snapshot_from_full')
    if args.clock != 'none':
        clock_path = Path(args.clock)
    cfg = get_config().parse_args([])
    cfg.lattice_size = [args.size] * 3; cfg.lattice_cu_nums = args.cu
    cfg.lattice_v_nums = 4; cfg.temperature = 600.; cfg.episode_length = args.events + 1
    cfg.history_len = 5; cfg.hidden_size = 256; cfg.layer_N = 2
    cfg.device = args.device; cfg.system_stats_interval = 5
    cfg.enable_energy_check = True; cfg.energy_check_interval = 10000
    cfg.use_recurrent_policy = True; cfg.use_wandb = False
    device = torch.device(args.device)
    if args.method in ('full', 'snapshot_from_full'):
        if not args.model:
            raise ValueError('trained model is required; no random-weight fallback')
        apply_checkpoint_config(cfg, args.model)
    if args.method.startswith('snapshot'):
        cfg.use_action_history = False; cfg.use_recurrent_memory = False
    seed_everything(args.seed, device)
    env = KMCEnvWrap(cfg); env.seed(args.seed); obs, _ = env.reset(); core = env.env
    initial_hash = hashlib.sha256(core.lattice.tobytes() + core.vacancy_pos.tobytes()).hexdigest()
    policy = None; checkpoint_hash = None
    if args.method in ('full', 'snapshot_from_full'):
        policy = R_MAPPOPolicy(cfg, env.action_space, device=device)
        checkpoint = Path(args.model) / 'actor_agent.pt'
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        policy.actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        policy.actor.eval(); policy.critic.eval()
    # Policy construction must not change the simulation stream.
    seed_everything(args.seed + 100000, device)
    clock_rng = np.random.default_rng(args.seed + 200000)
    clusters = CuClusters(core.cu_pos, core.lattice.shape, core.NN1)
    hit2 = clusters.vacancy_contacts(core.vacancy_pos, 2)
    hit4 = clusters.vacancy_contacts(core.vacancy_pos, 4)
    energy = np.empty(args.events + 1); structure = np.empty((args.events + 1, 2))
    dynamics = np.empty((args.events + 1, 3)); elapsed = np.zeros(args.events + 1)
    events = np.empty(args.events, dtype=np.int16); rates_log = np.empty((args.events, 2))
    energy[0] = core.energy_last
    initial_pairs = clusters.pairs
    initial_free = max(1, sum(len(clusters.adj[i]) == 0 for i in range(len(clusters.adj))))
    def record(step):
        fraction, maximum, count, largest = clusters.metrics()
        structure[step] = fraction, maximum
        dynamics[step] = np.clip(2 * (clusters.pairs - initial_pairs) / initial_free, 0, 1), count, largest
    record(0)
    state = np.zeros((4, cfg.recurrent_N, cfg.hidden_size), dtype=np.float32)
    masks = np.ones((4, 1), dtype=np.float32)
    start = time.monotonic()
    clock_x = np.empty((args.events,32,3),dtype=np.float32)
    clock_legal = np.empty((args.events,32),dtype=bool)
    from .residence_clock import inputs as clock_inputs
    for step in range(1, args.events + 1):
        rates = env.get_base_rates_np(copy=False)
        features = env.get_event_features_np(copy=False)
        clock_x[step-1], clock_legal[step-1] = clock_inputs(features, cfg.temperature)
        total = float(np.asarray(rates, dtype=np.float64).sum())
        if total <= 0:
            raise RuntimeError('nonabsorbing dead state; incomplete trajectory')
        if policy is None:
            mode = 'traditional' if args.method == 'classical' else args.method
            action, valid = sample_heuristic_action(mode, rates, features)
            if not valid: raise RuntimeError('no legal action')
        else:
            tokens, token_masks = env.get_sparse_defect_context_np(copy=False)
            if args.method.startswith('snapshot'):
                obs = obs.copy(); obs[..., 50:] = -1.; state.fill(0.)
            action, state, _ = sample_rl_action(policy, obs, state, masks, rates, features,
                                               tokens, token_masks, deterministic=False)
        event = int(np.asarray(action).reshape(-1)[0]); vac, direction = divmod(event, 8)
        old_v = tuple(map(int, core.vacancy_pos[vac]))
        old_cu = tuple(map(int, (core.vacancy_pos[vac] + core.NN1[direction]) % core.lattice.shape))
        is_cu = int(core.lattice[old_cu]) == 1
        selected_rate = float(np.asarray(rates).reshape(-1)[event])
        if selected_rate <= 0: raise RuntimeError('policy selected forbidden event')
        if args.method == 'classical':
            elapsed[step] = elapsed[step - 1] + clock_rng.exponential(1. / total)
        obs, _, _, _, done, info = env.step(action)
        if is_cu: clusters.move(old_cu, old_v)
        hit2 |= clusters.vacancy_contacts(core.vacancy_pos, 2)
        hit4 |= clusters.vacancy_contacts(core.vacancy_pos, 4)
        energy[step] = core.energy_last; record(step)
        events[step - 1] = event; rates_log[step - 1] = selected_rate, total
        if bool(np.asarray(done).reshape(-1)[0]):
            raise RuntimeError(f'early termination at {step}; cannot report matched budget')
        if step % 1000 == 0:
            print(json.dumps({'method': args.method, 'seed': args.seed, 'step': step,
                              'events_per_second': step / (time.monotonic() - start)}), flush=True)
    full_energy = float(core.calculate_system_energy())
    energy_drift = abs(full_energy - energy[-1])
    if energy_drift > cfg.energy_check_tol: raise RuntimeError('final energy consistency failed')
    name = f'cu{args.cu}_{args.method}_seed{args.seed}'
    exact_mean_residence = 1./rates_log[:,1]
    predicted_mean_residence = exact_mean_residence.copy()
    clock_report = {'conditional_residence_validated': True, 'source':'exact physical local rates'}
    if clock_path is not None and learned_clock:
        from .residence_clock import load as load_clock, domain as clock_domain
        clock, clock_report = load_clock(clock_path, clock_domain(core))
        with torch.no_grad():
            predicted_mean_residence = np.concatenate([clock(clock_x[i:i+2048],clock_legal[i:i+2048]).numpy()
                                                       for i in range(0,args.events,2048)])
    clock_log_error = np.log(predicted_mean_residence / exact_mean_residence)
    if not np.isfinite(clock_log_error).all(): raise ArithmeticError('nonfinite clock predictions')
    trajectory = output / (name + '.npz')
    np.savez_compressed(trajectory, energy_eV=energy, structure=structure, dynamics=dynamics,
                        elapsed_s=elapsed if args.method == 'classical' else np.full_like(elapsed, np.nan),
                        events=events, selected_and_total_rates=rates_log,
                        clock_features=clock_x, clock_legal=clock_legal,
                        exact_mean_residence_s=exact_mean_residence,
                        predicted_mean_residence_s=predicted_mean_residence,
                        clock_log_error=clock_log_error,
                        arrival_hit2=hit2, arrival_hit4=hit4)
    result = {'data_kind': 'raw_kmc_simulation', **vars(args), 'initial_state_sha256': initial_hash,
              'checkpoint_sha256': checkpoint_hash, 'trajectory': trajectory.name,
              'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
              'energy_auc_eV': float(np.trapezoid(energy[0] - energy, dx=1. / args.events)),
              'arrival_hit': float(hit2.mean()), 'arrival_hit_min4': float(hit4.mean()),
              'clock_error': float(np.abs(clock_log_error).mean()) if clock_path is not None else None,
              'clock_error_kind': 'conditional_mean_residence_mean_abs_log_ratio',
              'clock_error_max_abs_log_ratio': float(np.abs(clock_log_error).max()),
              'clock_relative_mean_residence_error_max': float(np.abs(np.expm1(clock_log_error)).max()),
              'clock_one_event_poisson_residual_rmse': float(np.sqrt(np.mean(np.expm1(clock_log_error)**2))),
              'clock_checkpoint': str(clock_path) if clock_path is not None and learned_clock else None,
              'clock_checkpoint_sha256': hashlib.sha256(clock_path.read_bytes()).hexdigest() if clock_path is not None and learned_clock else None,
              'clock_validation': clock_report,
              'arrival_definition': 'ever 1NN vacancy contact with a Cu component of size >=2, including initial state; >=4 sensitivity retained',
              'physical_clock_validated': args.method == 'classical', 'final_energy_drift_eV': energy_drift,
              'wall_seconds': time.monotonic() - start,
              'scope': 'current illustrative BCC pair parameter model; not validated Fe-Cu material parameters'}
    (output / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)



def main():
    from .train_bcc import positive_int
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=METHODS, default='full')
    parser.add_argument('--model')
    parser.add_argument('--cu', type=positive_int, required=True)
    parser.add_argument('--size', type=positive_int, default=50)
    parser.add_argument('--events', type=positive_int, required=True)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--clock', required=True, help='residence checkpoint path; none skips learned calibration')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__': main()
