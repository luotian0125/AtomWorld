"""Event-budget acceleration to each paired KMC trajectory's final energy."""
import json
from pathlib import Path
import numpy as np

METHOD_LABELS = {'classical':'Classical KMC', 'rate_greedy':'Rate-greedy KMC',
                 'energy_greedy':'Energy-greedy KMC', 'full':'AtomWorld (Full)',
                 'snapshot_independent':'Snapshot (separate training)',
                 'snapshot_from_full':'Snapshot (Full checkpoint)'}


def first_passage(energy, target_energy, reference_events):
    """Index zero is the initial state; no-hit trajectories are right censored."""
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or len(energy) < 2 or not np.isfinite(energy).all():
        raise ValueError('energy must contain finite initial and event states')
    if not np.isfinite(target_energy) or reference_events <= 0:
        raise ValueError('finite target and positive reference budget required')
    hits = np.flatnonzero(energy <= target_energy)
    step = int(hits[0]) if len(hits) else None
    return {'target_energy_eV': float(target_energy), 'reference_events': int(reference_events),
            'evaluation_events': len(energy) - 1, 'first_hit_step': step,
            'status': 'right_censored' if step is None else
                      ('initial_target_satisfied' if step == 0 else 'reached'),
            'af': float(reference_events / step) if step is not None and step > 0 else None,
            'af_upper_bound': float(reference_events / (len(energy) - 1)) if step is None else None}


def aggregate(records):
    """Mean/SE only over finite observed AFs, always accompanied by hit counts."""
    values = np.array([r['af'] for r in records if r['af'] is not None], dtype=float)
    return {'af_mean_reached': float(values.mean()) if len(values) else None,
            'af_se_reached': float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            'af_finite_count': len(values), 'af_reached_count': sum(r['first_hit_step'] is not None for r in records),
            'af_initial_satisfied_count': sum(r['first_hit_step'] == 0 for r in records),
            'af_censored_count': sum(r['first_hit_step'] is None for r in records),
            'af_total_count': len(records), 'af_aggregation': 'conditional mean/SE of finite reached trajectories'}


def paired_records(root, records, events, size):
    root = Path(root); output = {}
    for key, row in records.items():
        cu, method, seed = key
        reference = records.get((cu, 'classical', seed))
        if reference is None:
            raise RuntimeError(f'AF requires paired KMC reference for {key}')
        if row['initial_state_sha256'] != reference['initial_state_sha256']:
            raise RuntimeError('AF paired initial states differ')
        if any(r['events'] != events or r['size'] != size for r in (row, reference)):
            raise RuntimeError('AF requires matched evaluation budgets and lattice sizes')
        with np.load(root / row['trajectory']) as a, np.load(root / reference['trajectory']) as b:
            energy, ref_energy = a['energy_eV'], b['energy_eV']
            if len(energy) != events + 1 or len(ref_energy) != events + 1:
                raise RuntimeError('AF energy trajectory must include every event and initial state')
            value = first_passage(energy, ref_energy[-1], events)
        output[key] = {'cu': cu, 'method': method, 'seed': seed,
                       'method_trajectory_sha256': row['trajectory_sha256'],
                       'reference_trajectory_sha256': reference['trajectory_sha256'], **value}
    return output


def write_records(root, records):
    payload = {'definition': 'N_KMC / min{n >= 0 : E_method[n] <= E_paired_KMC[N_KMC]}',
               'comparison': 'stored energies, <=, zero additional tolerance',
               'censoring': 'not reached: AF < N_KMC/N_eval; never imputed as zero or one',
               'initial_target': 'n=0: AF undefined; count separately; never divide by zero',
               'aggregation': 'mean +/- SE over finite reached seeds, report reached/total and initial count',
               'scope': 'microscopic-event budget acceleration; not wall-clock or physical-time acceleration',
               'records': [records[key] for key in sorted(records)]}
    (Path(root) / 'af_records.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def format_af(row):
    if row['af_mean_reached'] is None:
        return 'N/A'
    error = row['af_se_reached']
    return f"{row['af_mean_reached']:.3f} ± {error:.3f}" if error is not None else f"{row['af_mean_reached']:.3f} (one finite hit)"

