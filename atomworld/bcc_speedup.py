"""Speedup diagnostics with matched targets and explicit censoring/timing scope."""
import numpy as np


def paired_speedup(reference, method, reference_wall, method_wall):
    for key in ('reference_events', 'target_energy_eV', 'evaluation_events'):
        if reference[key] != method[key]:
            raise ValueError('speedup requires the same target and evaluation budget')
    if not np.isfinite([reference_wall, method_wall]).all() or min(reference_wall, method_wall) <= 0:
        raise ValueError('positive finite evaluation times required')
    before, after = reference['first_hit_step'], method['first_hit_step']
    budget = method['evaluation_events']
    finite = before is not None and before > 0 and after is not None and after > 0
    return {'kmc_first_hit_step': before, 'method_first_hit_step': after,
            'first_passage_step_speedup': float(before / after) if finite else None,
            'first_passage_step_speedup_upper_bound': float(before / budget)
                if before is not None and before > 0 and after is None else None,
            'kmc_capped_first_passage_steps': before if before is not None else budget,
            'method_capped_first_passage_steps': after if after is not None else budget,
            'kmc_evaluation_wall_seconds': float(reference_wall),
            'method_evaluation_wall_seconds': float(method_wall),
            'fixed_budget_evaluation_wall_speedup': float(reference_wall / method_wall),
            'method_status': method['status']}


def aggregate(pairs):
    values = np.array([r['first_passage_step_speedup'] for r in pairs
                       if r['first_passage_step_speedup'] is not None], dtype=float)
    numerator = np.array([r['kmc_capped_first_passage_steps'] for r in pairs], dtype=float)
    denominator = np.array([r['method_capped_first_passage_steps'] for r in pairs], dtype=float)
    if not len(pairs) or denominator.mean() <= 0:
        raise ValueError('nontrivial target required for aggregate speedup')
    wall_ref = np.array([r['kmc_evaluation_wall_seconds'] for r in pairs])
    wall_method = np.array([r['method_evaluation_wall_seconds'] for r in pairs])
    return {'first_passage_step_speedup_mean_reached': float(values.mean()) if len(values) else None,
            'first_passage_step_speedup_se_reached': float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            'finite_pair_count': len(values), 'total_pair_count': len(pairs),
            'method_reached_count': sum(r['method_first_hit_step'] is not None for r in pairs),
            'method_initial_satisfied_count': sum(r['method_first_hit_step'] == 0 for r in pairs),
            'restricted_mean_first_passage_step_ratio': float(numerator.mean()/denominator.mean()),
            'fixed_budget_evaluation_wall_speedup': float(wall_ref.mean()/wall_method.mean()),
            'kmc_evaluation_wall_seconds_mean': float(wall_ref.mean()),
            'method_evaluation_wall_seconds_mean': float(wall_method.mean())}

