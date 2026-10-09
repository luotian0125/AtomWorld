"""Independent one-event Poisson residence calibration.

Target: absorption immediately after the next physical jump. Its backward
equation is Gamma_total * mean_residence = 1. This is distinct from a global
first-passage potential and does not repair a biased jump chain.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .adapters import make_material

FEATURE_SCHEMA = 'all_legal_events_Fe_Cu_types_and_exact_deltaE_over_kBT_v1'


def inputs(event_features, temperature=600.):
    """Exclude known rates, log rates, barriers and actor latent features."""
    f = np.asarray(event_features, dtype=np.float64).reshape(-1, event_features.shape[-1])
    x = f[:, [1, 2, 3]].copy()
    x[:, 2] /= 8.617333262145e-5 * temperature
    return x, f[:, 0] > .5


class ResidenceClock(nn.Module):
    """Learn a physical Arrhenius barrier from species and swap energy.

Three independently fitted parameters, with hard nonnegative barriers and
sum over the complete legal event catalogue. No actor or reward is imported.
"""
    def __init__(self, attempt_frequency=1e13):
        super().__init__()
        self.barrier = nn.Linear(3, 1, bias=False).double()
        with torch.no_grad():
            self.barrier.weight.copy_(torch.tensor([[8., 9., .1]], dtype=torch.float64))
        self.register_buffer('log_nu', torch.tensor(math.log(attempt_frequency), dtype=torch.float64))

    def log_total_rate(self, x, legal):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.log_nu.device)
        legal = torch.as_tensor(legal, dtype=torch.bool, device=x.device)
        if x.shape[-1] != 3 or legal.shape != x.shape[:-1] or not bool(legal.any(-1).all()):
            raise ValueError('clock needs a complete nonempty legal event catalogue')
        log_relative_rates = -self.barrier(x).squeeze(-1).clamp_min(0.)
        return self.log_nu + torch.logsumexp(log_relative_rates.masked_fill(~legal, -torch.inf), dim=-1)

    def forward(self, x, legal):
        return torch.exp(-self.log_total_rate(x, legal))


def domain(kernel):
    arrays = [np.asarray(kernel.pair_energies), np.asarray(kernel.E_a0)]
    return {'material': 'bcc', 'size': list(kernel.size), 'temperature_K': float(kernel.args.temperature),
            'cu_count': len(kernel.cu_pos), 'vacancies': len(kernel.vacancy_pos),
            'attempt_frequency': float(getattr(kernel.args, 'attempt_frequency', 1e13)),
            'rate_parameters_sha256': hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest(),
            'feature_schema': FEATURE_SCHEMA, 'target': 'absorb_after_next_physical_event',
            'scope': 'conditional mean residence; biased-policy physical evolution not certified'}


def collect(path, *, size, cu, episode, rounds, seed, stride=10):
    rng = np.random.default_rng(seed + 400000)
    adapter = make_material('bcc', size=(size,)*3, solutes=cu, vacancies=4, seed=seed, temperature=600)
    core = adapter.kernel
    rows = []; masks = []; log_rates = []; sampled_events = []; signatures = []
    started = time.monotonic()
    for run in range(rounds):
        reset_seed = seed + run
        state = np.random.get_state()
        try:
            np.random.seed(reset_seed); core.reset()
        finally:
            np.random.set_state(state)
        signatures.append(hashlib.sha256(core.lattice.tobytes() + core.vacancy_pos.tobytes()).hexdigest())
        for step in range(episode):
            rates = adapter.rates(); gamma = rates.sum()
            if gamma <= 0: raise RuntimeError('dead state in passive clock collection')
            if step % stride == 0:
                x, legal = inputs(core.get_event_features(copy=False), core.args.temperature)
                if not np.array_equal(legal, rates > 0): raise RuntimeError('event mask/rate mismatch')
                rows.append(x.astype(np.float32)); masks.append(legal); log_rates.append(float(np.log(gamma)))
            event = int(rng.choice(len(rates), p=rates/gamma)); sampled_events.append(event)
            adapter.apply(event)
        print(json.dumps({'clock_collection_round': run+1, 'rounds': rounds, 'seed': seed,
                          'passive_events': (run+1)*episode, 'wall_seconds': time.monotonic()-started}), flush=True)
    metadata = {'data_kind': 'raw_kmc_simulation', 'sampler': 'passive_physical_embedded_chain',
                'initial_seed': seed, 'reset_seeds': list(range(seed,seed+rounds)),
                'initial_state_sha256': signatures, 'domain': domain(core),
                'visited_events': rounds*episode, 'catalogue_sample_stride': stride,
                'labels': 'exact conditional mean from sum of ALL original physical rates; no noisy dwell labels'}
    temporary = Path(str(path)+'.tmp.npz')
    np.savez_compressed(temporary, x=np.asarray(rows), legal=np.asarray(masks), log_total_rate=np.asarray(log_rates),
                        passive_events=np.asarray(sampled_events,dtype=np.int16), metadata=np.asarray(json.dumps(metadata)))
    temporary.replace(path)


def read(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: torch.as_tensor(data[key], dtype=torch.bool if key=='legal' else torch.float64)
                  for key in ('x', 'legal', 'log_total_rate')}
        metadata = json.loads(str(data['metadata']))
    if metadata['data_kind'] != 'raw_kmc_simulation' or not result['legal'].any(-1).all():
        raise ValueError('invalid clock dataset')
    return result, metadata


@torch.no_grad()
def validate(model, data):
    error = data['log_total_rate'] - model.log_total_rate(data['x'], data['legal'])
    relative = torch.expm1(error).abs()
    residual = torch.expm1(error)
    return {'clock_error_mean_abs_log_ratio': float(error.abs().mean()),
            'relative_mean_residence_error_max': float(relative.max()),
            'one_event_poisson_residual_rmse': float(residual.square().mean().sqrt()),
            'conditional_residence_validated': bool(error.abs().mean() <= .01 and relative.max() <= .05),
            'policy_physical_time_validated': False, 'states': len(error)}


def fit(train_path, validation_path, checkpoint, iterations=250):
    train, train_meta = read(train_path); val, val_meta = read(validation_path)
    if train_meta['domain'] != val_meta['domain'] or set(train_meta['reset_seeds']) & set(val_meta['reset_seeds']):
        raise ValueError('clock domain mismatch or leaked train/validation seeds')
    torch.manual_seed(781)
    model = ResidenceClock(train_meta['domain']['attempt_frequency'])
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=iterations,
                                  line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-16)
    def closure():
        optimizer.zero_grad()
        log_residual = train['log_total_rate'] - model.log_total_rate(train['x'], train['legal'])
        loss = log_residual.square().mean()
        if not torch.isfinite(loss): raise ArithmeticError('nonfinite independent residence fit')
        loss.backward(); return loss
    optimizer.step(closure); model.eval()
    report = validate(model, val)
    report['training'] = validate(model, train)
    report['clock_error_definition'] = 'mean abs(log(predicted_mean_residence / exact_mean_residence))'
    report['claim'] = 'one-event Poisson residence calibration only; not global MFPT or policy physical time'
    payload = {'schema':1, 'kind':'independent_one_event_poisson_residence', 'domain':train_meta['domain'],
               'state_dict':model.state_dict(), 'validation':report,
               'train_metadata':train_meta, 'validation_metadata':val_meta,
               'dataset_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name,path in (('train',train_path),('validation',validation_path))}}
    temporary = Path(str(checkpoint)+'.tmp')
    torch.save(payload, temporary); temporary.replace(checkpoint)
    Path(checkpoint).with_suffix('.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='state_dict'},indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    return model, report


def load(path, expected_domain):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload['kind'] != 'independent_one_event_poisson_residence' or payload['domain'] != expected_domain:
        raise ValueError('residence clock checkpoint/domain mismatch')
    if not payload['validation'].get('conditional_residence_validated', False):
        raise ValueError('residence clock checkpoint did not pass validation')
    model = ResidenceClock(expected_domain['attempt_frequency'])
    model.load_state_dict(payload['state_dict'],strict=True); model.eval()
    return model,payload['validation']


def ensure(root, *, cu, episode, size=50, rounds=25):
    """Shared checkpoint per physical condition. POSIX lock handles paired workers."""
    import fcntl
    out = Path(root)/'residence-clocks'/f'ep{episode}_cu{cu}'; out.mkdir(parents=True,exist_ok=True)
    checkpoint = out/'clock.pt'
    with (out/'fit.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not checkpoint.exists():
            train = out/'train.npz'; val = out/'validation.npz'
            if not train.exists(): collect(train,size=size,cu=cu,episode=episode,rounds=rounds,seed=501,stride=10)
            if not val.exists(): collect(val,size=size,cu=cu,episode=1000,rounds=4,seed=601,stride=2)
            fit(train,val,checkpoint)
    return checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--cu',type=int,required=True)
    p.add_argument('--episode',type=int,required=True);p.add_argument('--size',type=int,default=50)
    p.add_argument('--rounds',type=int,default=25);args=p.parse_args();torch.set_num_threads(1)
    print(ensure(args.root,cu=args.cu,episode=args.episode,size=args.size,rounds=args.rounds),flush=True)


if __name__=='__main__':main()
