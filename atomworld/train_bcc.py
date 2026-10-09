"""Train the Full BCC AtomWorld policy with an independent calibration layer."""
import argparse
import json
from pathlib import Path
import torch


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=positive_int, required=True, help='events per training round')
    p.add_argument('--cu', type=positive_int, required=True, help='number of Cu atoms')
    p.add_argument('--size', type=positive_int, default=50, help='unit cells per axis')
    p.add_argument('--tag', default='atomworld')
    p.add_argument('--rounds', type=positive_int, default=25)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        p.error('tag must be a directory name')
    if args.cu + 4 > 2 * args.size ** 3:
        p.error('Cu atoms and four vacancies exceed the lattice size')
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('a CUDA device is required')
    from .backends.bcc.train import main as train
    experiment = f'{args.tag}_ep{args.episode}_cu{args.cu}_full'
    from .paths import training_root
    runroot = training_root('bcc') / 'KMCEnv' / 'single_event_kmc' / 'rmappo' / experiment
    if runroot.exists(): raise FileExistsError('fresh training cannot overwrite an existing run')
    episode = 256 if args.smoke else args.episode
    rounds = 1 if args.smoke else args.rounds
    command = ['--algorithm_name', 'rmappo', '--scenario_name', 'single_event_kmc',
               '--experiment_name', experiment, '--lattice_size', *([str(args.size)] * 3),
               '--lattice_v_nums', '4', '--lattice_cu_nums', str(args.cu), '--temperature', '600',
               '--episode_length', str(episode), '--num_env_steps', str(episode * rounds),
               '--hidden_size', '256', '--layer_N', '2', '--history_len', '5', '--device', 'cuda:0',
               '--seed', str(args.seed), '--no_wandb', '--system_stats_interval', '5',
               '--enable_energy_check', '--energy_check_interval', '1000',
               '--ppo_epoch', '8', '--num_mini_batch', '4', '--data_chunk_length', '32',
               '--gamma', '0.999', '--gae_lambda', '0.97', '--lr', '0.0003', '--critic_lr', '0.0003',
               '--entropy_coef', '0.002', '--plot_interval', '0', '--log_interval', '1', '--save_interval', '1',
               '--policy_logit_scale_start', '0.35', '--policy_logit_scale_end', '1.10',
               '--policy_logit_scale_schedule', 'linear', '--rate_prior_scale_start', '1.0',
               '--rate_prior_scale_end', '0.35', '--rate_prior_scale_schedule', 'linear',
               '--energy_prior_scale_start', '0.20', '--energy_prior_scale_end', '1.10',
               '--energy_prior_scale_schedule', 'linear', '--cu_move_bonus', '0.06',
               '--fe_move_penalty', '-0.004', '--step_penalty', '-0.002', '--new_best_energy_bonus', '0.08',
               '--aux_phys_teacher_coef', '0.60', '--aux_event_time_coef', '0',
               '--aux_teacher_rate_weight', '0.30', '--aux_teacher_cu_bonus', '0.20',
               '--aux_teacher_fe_penalty', '0.08']
    print(json.dumps({'fresh_training': vars(args), 'argv': command,
                      'checkpoint_selection': 'final fixed-budget checkpoint; no evaluation-seed selection',
                      'clock_training': 'disabled in policy optimization; fitted independently'}), flush=True)
    train(command)
    checkpoint = runroot / 'run1' / 'models'
    metadata = {'parameters': vars(args), 'argv': command, 'checkpoint': str(checkpoint),
                'completed_rounds': rounds, 'microscopic_events': episode * rounds,
                'publication_material_parameters_validated': False}
    (runroot / 'training_complete.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
