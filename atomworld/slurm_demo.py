"""Train independent Poisson networks on exact, tiny MATERIAL state graphs.

Validation starts from an independent material seed and uses the same fixed
target/domain. These are finite-domain checks, not unseen-physics generalization.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapters import make_material
from .cli import domain_metadata, site_target
from .dataset import enumerate_graph, save_rows
from .network import fit


def graph_rows(graph):
    tau, _ = graph.solve()
    rows=[]
    for i in range(len(tau)):
        rows.append({"x":graph.features[i],"absorbing":graph.absorbing[i],"rates":graph.rates[i],
                     "successor_x":graph.features[graph.successors[i]],
                     "successor_absorbing":graph.absorbing[graph.successors[i]],
                     "reference_mean_s":tau[i],"reference_se_s":0.,"reference_censored":0})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    reports={}
    for material in ('bcc','fcc','hcp'):
        adapter=make_material(material,solutes=1 if material=='hcp' else 0,seed=1)
        target=site_target(adapter);domain=domain_metadata(adapter,target)
        for split,seed in (('train',1),('validation',2)):
            a=make_material(material,solutes=1 if material=='hcp' else 0,seed=seed)
            graph=enumerate_graph(a,target)
            meta={'data_kind':'raw_kmc_simulation','domain':domain,'initial_seed':seed,
                  'reference_kind':'exhaustive_exact_generator','coverage':'full_tiny_state_space'}
            save_rows(output.parent/f'{material}_{split}.npz',graph_rows(graph),meta)
        _,report=fit(output.parent/f'{material}_train.npz',output.parent/f'{material}_validation.npz',
                     output.parent/f'{material}_potential.pt',hidden=64,iterations=1000)
        report['claim']='tiny finite-domain network check; not paper-scale or policy clock validation'
        reports[material]=report
        print(json.dumps({'material':material,**report}),flush=True)
    output.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    if not all(r['mfpt_reference_validated'] for r in reports.values()):
        raise SystemExit(2)


if __name__=='__main__':main()
