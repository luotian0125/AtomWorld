"""Independent Fe-Cu finite-generator audit, including the signed-clock issue."""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapters import make_material
from .cli import domain_metadata, site_target
from .dataset import enumerate_graph, save_rows
from .slurm_demo import graph_rows
from .network import fit
from .poisson import FiniteRateGraph


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    adapter = make_material('bcc', solutes=1, seed=71, temperature=600)
    target = site_target(adapter); domain = domain_metadata(adapter, target)
    graphs = []
    for split, seed in (('train', 71), ('validation', 83)):
        material = make_material('bcc', solutes=1, seed=seed, temperature=600)
        graph = enumerate_graph(material, target); graphs.append(graph)
        save_rows(root / (split + '.npz'), graph_rows(graph),
                  {'data_kind': 'raw_kmc_simulation', 'domain': domain, 'initial_seed': seed,
                   'reference_kind': 'exhaustive_exact_generator', 'coverage': 'full_tiny_alloy_state_space'})
    graph = graphs[0]; tau, second = graph.solve()
    residual = float(np.max(np.abs(graph.residuals(tau))))
    scaled = FiniteRateGraph([r * 1.17 for r in graph.rates], graph.successors, graph.absorbing, graph.features)
    tau_scaled, _ = scaled.solve()
    scaling_error = float(np.max(np.abs(tau_scaled - tau / 1.17)) / np.max(tau))
    rng = np.random.default_rng(90217)
    draws = np.asarray([graph.sample_hitting_time(0, rng) for _ in range(10000)])
    se = float(draws.std(ddof=1) / np.sqrt(len(draws)))
    signed = []; expectation_error = []
    for i, rates in enumerate(graph.rates):
        if graph.absorbing[i]: continue
        progress = tau[i] - tau[graph.successors[i]]; signed.extend(progress.tolist())
        expectation_error.append(abs(np.dot(rates / rates.sum(), progress) * rates.sum() - 1))
    _, network = fit(root / 'train.npz', root / 'validation.npz', root / 'potential.pt',
                     hidden=128, iterations=2000)
    report = {'states': len(tau), 'domain': domain, 'max_backward_residual': residual,
              'rate_scaling_relative_error': scaling_error, 'exact_mfpt_s': float(tau[0]),
              'passive_mean_s': float(draws.mean()), 'passive_se_s': se,
              'samples': len(draws), 'passive_within_5se': bool(abs(draws.mean() - tau[0]) <= 5 * se),
              'signed_progress_negative_count': int(np.sum(np.asarray(signed) < 0)),
              'signed_progress_count': len(signed),
              'passive_conditional_expectation_max_error': float(max(expectation_error)),
              'network': network, 'paper_scale_validated': False, 'policy_physical_time_validated': False,
              'scope': '2^3 cells, one Cu and one vacancy, fixed-site hitting target. Finite domain only; independent construction seeds cover the same complete graph.'}
    (root / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if residual > 1e-7 or scaling_error > 1e-7 or not report['passive_within_5se'] or not network['mfpt_reference_validated']:
        raise SystemExit(2)


if __name__ == '__main__': main()
