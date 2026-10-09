import argparse

def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default="rmappo", choices=["rmappo", "mappo", "ippo"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.set_defaults(cuda=True, cuda_deterministic=True)
    parser.add_argument("--cuda", dest="cuda", action='store_true', help="Use CUDA for training/evaluation when available.")
    parser.add_argument("--no_cuda", dest="cuda", action='store_false', help="Force CPU execution even if CUDA is available.")
    parser.add_argument("--cuda_deterministic", dest="cuda_deterministic",
                        action='store_true', help="Enable deterministic CUDA behavior for reproducibility.")
    parser.add_argument("--no_cuda_deterministic", dest="cuda_deterministic",
                        action='store_false', help="Allow nondeterministic CUDA kernels for potentially higher throughput.")
    parser.add_argument("--num_env_steps", type=int, default=10_000_000,
                        help='Number of environment steps to train (default: 10e6)')
    parser.set_defaults(use_wandb=False)
    parser.add_argument("--use_wandb", dest="use_wandb", action='store_true', help="Log training/evaluation metrics to Weights & Biases.")
    parser.add_argument("--no_wandb", dest="use_wandb", action='store_false', help="Disable Weights & Biases and use local TensorBoard logging instead.")
    parser.add_argument("--wandb_project", type=str, default="AtomWorld-Mem", help="Weights & Biases project name used when --use_wandb is set.")
    parser.add_argument("--wandb_entity", type=str, default="", help="Optional Weights & Biases entity used when --use_wandb is set.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='KMCEnv', help="specify the name of environment")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.set_defaults(use_centralized_V=True)
    parser.add_argument("--use_centralized_V", dest="use_centralized_V", action='store_true',
                        help="Use a centralized critic/value function.")
    parser.add_argument("--disable_centralized_V", dest="use_centralized_V", action='store_false',
                        help="Disable the centralized critic/value function.")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--inference_cache_size", type=int, default=128,
                        help="Maximum frozen dense keyframes cached at inference; 0 disables caching.")
    parser.add_argument("--history_len", type=int, default=5,
                        help="Number of recent actions appended to each vacancy observation.")
    parser.add_argument("--dense_local_scale", type=float, default=1.0,
                        help="Scale applied to dense local vacancy features for context ablations.")
    parser.set_defaults(use_sparse_context=True, use_action_history=True, use_recurrent_memory=True)
    parser.add_argument("--use_sparse_context", dest="use_sparse_context", action="store_true",
                        help="Use sparse multi-scale defect context in the actor.")
    parser.add_argument("--disable_sparse_context", dest="use_sparse_context", action="store_false",
                        help="Disable sparse multi-scale defect context in the actor.")
    parser.add_argument("--use_action_history", dest="use_action_history", action="store_true",
                        help="Use short-term action history in actor/critic inputs.")
    parser.add_argument("--disable_action_history", dest="use_action_history", action="store_false",
                        help="Zero the short-term action history features.")
    parser.add_argument("--use_recurrent_memory", dest="use_recurrent_memory", action="store_true",
                        help="Use recurrent actor memory.")
    parser.add_argument("--disable_recurrent_memory", dest="use_recurrent_memory", action="store_false",
                        help="Disable recurrent actor memory for snapshot-policy baselines.")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.set_defaults(use_valuenorm=True)
    parser.add_argument("--use_valuenorm", dest="use_valuenorm", action='store_true', help="Use running mean/std normalization for value targets.")
    parser.add_argument("--disable_valuenorm", dest="use_valuenorm", action='store_false', help="Disable value normalization.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    
    # 【优化 1】：缩短 BPTT 初始切片长度，加快早期收敛
    parser.add_argument("--data_chunk_length", type=int, default=8,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    # 【优化 2】：下调学习率，稳定含有 Transformer 和 GRU 的复杂时空网络
    parser.add_argument("--lr", type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument("--spn_learning_rate", type=float, default=3e-4,
                        help='spn learning rate (default: 3e-4)')
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help='critic learning rate (default: 3e-4)')
                        
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
                        
    # 【优化 3】：下调探索熵系数，避免带记忆的网络陷入过度探索
    parser.add_argument("--entropy_coef", type=float, default=0.002,
                        help='entropy term coefficient (default: 0.002)')
                        
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
                        
    # 【优化 4】：极其关键的梯度裁剪收紧，防止 RNN 梯度爆炸导致训练崩盘
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
                        
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.set_defaults(use_huber_loss=True)
    parser.add_argument("--use_huber_loss", dest="use_huber_loss", action='store_true', help="Use Huber loss for value optimization.")
    parser.add_argument("--disable_huber_loss", dest="use_huber_loss", action='store_false', help="Disable Huber loss and fall back to MSE.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")
    parser.add_argument("--enable_episode_plots", action="store_true", default=True,
                        help="Enable per-episode plot dumping. Disabled by default for speed.")
    parser.add_argument("--plot_interval", type=int, default=1,
                        help="If >0 and plotting is enabled, save one plot every plot_interval episodes.")
    parser.add_argument("--enable_snapshot_panels", action="store_true", default=False,
                        help="Save 3D Cu clustering snapshot panels during training.")
    parser.add_argument("--snapshot_panel_interval", type=int, default=25,
                        help="If snapshot panels are enabled, save one panel every snapshot_panel_interval episodes.")
    parser.add_argument("--snapshot_panel_steps", type=int, nargs="*", default=None,
                        help="Episode step indices used for Cu snapshot panels. Defaults to four evenly spaced frames from 0 to episode_length.")
    parser.add_argument("--snapshot_panel_show_vacancy", action="store_true", default=False,
                        help="Also render vacancies in the Cu snapshot panels.")
    parser.add_argument("--snapshot_panel_output_root", type=str, default="",
                        help="Optional output root for Cu snapshot panels. Defaults to <run_dir>/snapshot_panels.")
    parser.add_argument("--show_collect_progress", action="store_true", default=True,
                        help="Show tqdm progress bars during rollout collection.")
    parser.add_argument("--show_eval_progress", action="store_true", default=True,
                        help="Show tqdm progress bars during evaluation.")
    parser.add_argument("--policy_logit_scale", type=float, default=0.20,
                        help="Residual scale for learned logits before combining with log(base_rates).")
    parser.add_argument("--policy_logit_scale_start", type=float, default=None,
                        help="Starting scale for the learned-logit residual schedule. Defaults to policy_logit_scale.")
    parser.add_argument("--policy_logit_scale_end", type=float, default=None,
                        help="Final scale for the learned-logit residual schedule. Defaults to the start scale.")
    parser.add_argument("--policy_logit_scale_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Schedule used to update policy_logit_scale during training.")
    parser.add_argument("--rate_prior_scale", type=float, default=1.0,
                        help="Scale applied to log(base_rates) when fusing the learned event logits with the KMC prior.")
    parser.add_argument("--rate_prior_scale_start", type=float, default=None,
                        help="Starting scale for the KMC rate prior schedule. Defaults to rate_prior_scale.")
    parser.add_argument("--rate_prior_scale_end", type=float, default=None,
                        help="Final scale for the KMC rate prior schedule. Defaults to the start scale.")
    parser.add_argument("--rate_prior_scale_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Schedule used to update the KMC rate-prior strength during training.")
    parser.add_argument("--energy_prior_scale", type=float, default=0.0,
                        help="Scale applied to the explicit energy-gain prior when fusing event logits.")
    parser.add_argument("--energy_prior_scale_start", type=float, default=None,
                        help="Starting scale for the energy-gain prior schedule. Defaults to energy_prior_scale.")
    parser.add_argument("--energy_prior_scale_end", type=float, default=None,
                        help="Final scale for the energy-gain prior schedule. Defaults to the start scale.")
    parser.add_argument("--energy_prior_scale_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Schedule used to update the energy-gain prior strength during training.")
    parser.add_argument("--energy_prior_downhill_bonus", type=float, default=0.20,
                        help="Extra prior bias added to downhill events in the explicit energy prior.")
    parser.add_argument("--energy_prior_cu_bonus", type=float, default=0.10,
                        help="Extra prior bias added to Cu-mediated events in the explicit energy prior.")
    parser.add_argument("--energy_prior_fe_penalty", type=float, default=0.04,
                        help="Penalty applied to Fe-mediated events in the explicit energy prior.")
    parser.add_argument("--energy_prior_clip", type=float, default=3.0,
                        help="Absolute clip applied to the normalized explicit energy prior before scaling.")
    parser.set_defaults(use_direction_aware_head=True)
    parser.add_argument("--use_direction_aware_head", dest="use_direction_aware_head", action="store_true",
                        help="Use direction-aware cross-attention head for 8 jump directions.")
    parser.add_argument("--disable_direction_aware_head", dest="use_direction_aware_head", action="store_false",
                        help="Use the simpler per-direction scoring head.")
    parser.add_argument("--enable_auto_compare", dest="enable_auto_compare", action="store_true", default=False,
                        help="Run scripts/compare_methods.py automatically during training.")
    parser.add_argument("--disable_auto_compare", dest="enable_auto_compare", action="store_false",
                        help="Disable automatic compare during training.")
    parser.add_argument("--auto_compare_interval", type=int, default=25,
                        help="Run auto compare every N training episodes.")
    parser.add_argument("--auto_compare_workers", type=int, default=1, choices=[1, 2, 4],
                        help="Independent rollout processes for automatic comparison.")
    parser.add_argument("--auto_compare_require_success", action="store_true", default=False,
                        help="Stop training if a scheduled comparison fails.")
    parser.add_argument("--auto_compare_script", type=str, default="scripts/compare_methods.py",
                        help="Path to the fixed-budget comparison script.")
    parser.add_argument("--auto_compare_python_bin", type=str, default="",
                        help="Python executable used to launch the auto compare subprocess. Defaults to the current interpreter.")
    parser.add_argument("--auto_compare_output_root", type=str, default="",
                        help="Optional root directory for auto compare outputs. Defaults to <run_dir>/auto_compare.")
    parser.add_argument("--auto_compare_bench_step", type=int, default=-1,
                        help="Benchmark rollout length for auto compare. If <= 0, use episode_length.")
    parser.add_argument("--auto_compare_energy_series_interval_steps", type=int, default=50,
                        help="Sampling interval for auto compare CSV series.")
    parser.add_argument("--auto_compare_plot_band", type=str, default="stderr",
                        choices=["std", "stderr", "none"],
                        help="Plot band style used by auto compare.")
    parser.add_argument("--auto_compare_bootstrap_samples", type=int, default=2000,
                        help="Bootstrap samples used in auto compare summaries.")
    parser.add_argument("--auto_compare_seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Seed list used by auto compare.")
    parser.add_argument("--auto_compare_modes", type=str, nargs="+", default=["traditional", "rl4kmc_stochastic", "rl4kmc_greedy"],
                        help="Compare modes used by auto compare.")
    parser.add_argument("--auto_compare_target_energy_drops", type=float, nargs="*", default=[4.0, 5.0, 6.0],
                        help="Target energy-drop thresholds used by auto compare.")
    parser.add_argument("--auto_compare_rate_energy_beta", type=float, default=1.0,
                        help="Beta passed to the Rate x Energy heuristic baseline.")
    parser.add_argument("--enable_auto_paper_metrics", dest="enable_auto_paper_metrics", action="store_true", default=False,
                        help="Run build_paper_metrics.py automatically after each successful auto compare.")
    parser.add_argument("--disable_auto_paper_metrics", dest="enable_auto_paper_metrics", action="store_false",
                        help="Disable automatic paper-metric summarization after auto compare.")
    parser.add_argument("--auto_paper_metrics_script", type=str, default="scripts/build_paper_metrics.py",
                        help="Path to the build_paper_metrics.py script used after auto compare.")
    parser.add_argument("--auto_paper_metrics_reference_mode", type=str, default="traditional",
                        help="Reference mode passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_main_mode", type=str, default="rl4kmc_stochastic",
                        help="Main method mode passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_method_label", type=str, default="AtomWorld-Mem",
                        help="Method label passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_effective_energy_eps", type=float, default=1e-6,
                        help="Effective-transition threshold passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_fidelity_metrics", type=str, nargs="+",
                        default=["advancement_factor", "cluster_count", "largest_cluster_size"],
                        help="Dynamics-fidelity metrics passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_time_grid_points", type=int, default=200,
                        help="Time grid points passed to build_paper_metrics.py.")
    parser.add_argument("--auto_paper_metrics_output_prefix", type=str, default="paper_metrics",
                        help="Output prefix used by build_paper_metrics.py.")
    parser.add_argument("--enable_auto_main_table1", dest="enable_auto_main_table1", action="store_true", default=False,
                        help="Run scripts/build_main_table.py automatically after paper metrics.")
    parser.add_argument("--disable_auto_main_table1", dest="enable_auto_main_table1", action="store_false",
                        help="Disable automatic main Table 1 export.")
    parser.add_argument("--auto_main_table1_script", type=str, default="scripts/build_main_table.py",
                        help="Path to the compact table builder.")
    parser.add_argument("--auto_main_table1_output_prefix", type=str, default="main_table1",
                        help="Output prefix used by the compact table builder.")
    parser.add_argument("--auto_main_table1_target_drop", type=float, default=-1.0,
                        help="Target energy drop used for Table 1 steps-to-target speedup. If <=0, use the largest auto_compare target.")
    parser.add_argument("--direction_attention_heads", type=int, default=4,
                        help="Number of attention heads in the direction-aware action head.")
    parser.add_argument("--direction_attention_dropout", type=float, default=0.05,
                        help="Dropout in the direction-aware action head.")
    parser.add_argument("--reserve_other_vacancy_tokens", action="store_true", default=False,
                        help="Reserve sparse-context slots for other vacancies before nearby Cu atoms.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    parser.add_argument("--spn_model_dir", type=str, default=None, help="by default None. set the path to pretrained spn model.")
    parser.add_argument("--spn_data_path", type=str, default=None, help="by default None. set the path to load spn data.")
    
    

    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)

    
    parser.add_argument("--scenario_name", type=str,
                        default="academy_3_vs_1_with_keeper", 
                        help="which scenario to run on.")
    parser.add_argument("--representation", type=str, default="simple115v2", 
                        choices=["simple115v2", "extracted", "pixels_gray", 
                                 "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring", 
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")
    parser.add_argument("--remove_redundancy", action="store_true", 
                        default=False, 
                        help="by default False. If True, remove redundancy features")
    parser.add_argument("--zero_feature", action="store_true", 
                        default=False, 
                        help="by default False. If True, replace -1 by 0")
    parser.add_argument("--eval_deterministic", action="store_false", 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")

    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
    

    parser.add_argument("--gnn_attention_heads", type=int, default=4,
                        help="Number of attention heads in GNN.")
    parser.add_argument("--gnn_attention_dropout", type=float, default=0.1,
                        help="Dropout rate for attention in GNN.")
    parser.add_argument("--gnn_layers", type=int, default=3,
                        help="Number of GNN layers.")
    
    parser.add_argument("--temperature", type=float, default=663.0,
                        help="Temperature for KMC.")
    
    parser.add_argument("--lattice_size",
                        type=int,       # 关键：将每个字符串参数转换为整数
                        nargs=3,        # 关键：告诉argparse接收三个参数
                        default=[50, 50, 50])
    parser.add_argument("--neighbor_radius", type=float, default=4.0,
                        help="Neighbor radius for GNN static template graph.")
    parser.add_argument("--max_num_neighbors", type=int, default=64,
                        help="Max number of neighbors for GNN. Defaults high enough to avoid truncating the 4NN template graph.")
    parser.add_argument("--lattice_cu_nums", type=int, default=125,
                        help="Number of Cu atoms in the lattice.")
    parser.add_argument("--lattice_v_nums", type=int, default=4,
                        help="Number of V in the lattice.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Preferred torch device string, e.g. cpu, cuda:0, cuda:1.")

    parser.add_argument("--layer_N", type=int, default=10,
                        help="Number of layers for embed/actor/critic networks")
    parser.add_argument("--train_name", type=str, default="train",)
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Scale factor for rewards.")
    parser.add_argument("--spn_edge_dim", type=int, default=5,
                        help="Edge Dim for SPN networks")
    parser.add_argument("--event_feature_dim", type=int, default=9,
                        help="Number of physical descriptors exposed per candidate KMC event.")

    # 精细化奖励参数
    parser.add_argument("--cu_move_bonus", type=float, default=0.05, help="Reward for moving a Cu atom")
    parser.add_argument("--fe_move_penalty", type=float, default=-0.002, help="Penalty for moving a Fe atom")
    parser.add_argument("--step_penalty", type=float, default=-0.005, help="Time penalty per step")
    parser.add_argument("--illegal_action_penalty", type=float, default=-0.1, help="Penalty for invalid collision")
    parser.add_argument("--conflict_penalty", type=float, default=-0.05, help="Penalty for losing conflict resolution")
    parser.add_argument("--new_best_energy_bonus", type=float, default=0.05,
                        help="Bonus applied when the rollout reaches a new lowest system energy.")
    parser.add_argument("--best_event_gap_penalty_coef", type=float, default=0.12,
                        help="Penalty coefficient for choosing an event whose energy gain is worse than the best currently legal event.")
    parser.add_argument("--enable_energy_check", action='store_true', default=False, help="Enable physical energy conservation check")
    parser.add_argument("--energy_check_interval", type=int, default=200,
                        help="When energy checking is enabled, recompute full system energy every N env steps.")
    parser.add_argument("--energy_check_tol", type=float, default=1e-5,
                        help="Tolerance used when comparing incremental energy with the full recomputed energy.")
    parser.add_argument("--system_stats_interval", type=int, default=1,
                        help="Refresh expensive global system stats every N env steps. Use >1 to reduce collect overhead.")
    parser.add_argument("--aux_phys_teacher_coef", type=float, default=0.5,
                        help="Coefficient of the auxiliary physical teacher loss that biases policy updates toward stronger energy-reducing events.")
    parser.add_argument("--aux_event_time_coef", type=float, default=0.2,
                        help="Coefficient of the auxiliary event-time hazard loss that aligns the restored-state hazard field with KMC event probabilities and waiting times.")
    parser.add_argument("--aux_teacher_energy_weight", type=float, default=1.0,
                        help="Weight of per-event energy gain in the auxiliary physical teacher distribution.")
    parser.add_argument("--aux_teacher_rate_weight", type=float, default=0.35,
                        help="Weight of per-event log-rate in the auxiliary physical teacher distribution.")
    parser.add_argument("--aux_teacher_cu_bonus", type=float, default=0.15,
                        help="Teacher preference bonus for Cu-mediated jumps.")
    parser.add_argument("--aux_teacher_fe_penalty", type=float, default=0.05,
                        help="Teacher penalty for Fe-mediated jumps.")
    parser.add_argument("--aux_teacher_temperature", type=float, default=0.75,
                        help="Temperature used to soften the auxiliary physical teacher distribution.")

    return parser
