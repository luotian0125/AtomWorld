"""Reproducible calibration commands with explicit claim boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .adapters import make_material
from .dataset import collect_rows, enumerate_graph, save_rows
from .targets import Target, paper_target


def domain_metadata(adapter, target):
    k = adapter.kernel
    if adapter.material == "hcp":
        physics = {"parameters": k.parameters.metadata(), "energy": k.energy_model.metadata(),
                   "temperature_K": k.temperature_K, "size": list(k.lattice.size), "hydrogen_count": k.num_agents}
    else:
        # Hash actual numerical input arrays as well as descriptive metadata.
        arrays = [np.asarray(k.pair_energies), np.asarray(k.E_a0)]
        physics = {"temperature_K": float(k.args.temperature), "size": list(k.size),
                   "rate_parameters_sha256": hashlib.sha256(b"".join(x.tobytes() for x in arrays)).hexdigest(),
                   "parameters": getattr(k, "parameter_metadata", {"source": "legacy BCC pair model", "publication_ready": False}),
                   "attempt_frequency": getattr(k.args, "attempt_frequency", 1e13),
                   "vacancies": len(k.vacancy_pos), "solutes": len(k.cu_pos)}
    return {"material": adapter.material, "physics": physics,
            "target": {"kind": target.kind, "options": target.options},
            "feature_schema": "full_microstate_and_target_memory_v1",
            "claim": "MFPT potential, not a calibrated policy physical clock"}


def site_target(adapter):
    k = adapter.kernel
    if adapter.material == "hcp":
        source = int(k.lattice.hydrogen_sites[0])
        site = int(k.lattice.event_targets[source, 0])
    else:
        site = ((k.vacancy_pos[0] + k.NN1[0]) % k.lattice.shape).tolist()
    return Target("site", {"site": site})


def run_exact_smoke(output, samples=5000):
    """Real kernels, tiny periodic domains, explicit first-arrival boundary."""
    if samples < 2:
        raise ValueError("at least two Monte Carlo samples are required")
    reports = {}
    for material in ("bcc", "fcc", "hcp"):
        adapter = make_material(material, solutes=1 if material == "hcp" else 0)
        target = site_target(adapter)
        graph = enumerate_graph(adapter, target)
        tau, second = graph.solve()
        rng = np.random.default_rng(71001)
        draws = np.array([graph.sample_hitting_time(0, rng) for _ in range(samples)])
        se = float(draws.std(ddof=1) / np.sqrt(samples))
        error = abs(draws.mean() - tau[0])
        reports[material] = {"states": len(tau), "target": target.options,
            "exact_mfpt_s": float(tau[0]), "exact_variance_s2": float(second[0] - tau[0] ** 2),
            "passive_mean_s": float(draws.mean()), "passive_standard_error_s": se,
            "max_backward_residual": float(np.max(np.abs(graph.residuals(tau)))),
            "mc_within_5_standard_errors": bool(error <= 5 * se),
            "samples": samples, "domain": domain_metadata(adapter, target),
            "claim": "kernel/exact-solver verification only; not paper-scale network validation"}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not all(r["mc_within_5_standard_errors"] for r in reports.values()):
        raise RuntimeError("passive KMC and exact MFPT disagree")
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("verify"); smoke.add_argument("--output", default="results/exact_validation.json")
    smoke.add_argument("--samples", type=int, default=5000)
    collect = sub.add_parser("collect")
    collect.add_argument("--material", choices=["bcc", "fcc", "hcp"], required=True)
    collect.add_argument("--size", type=int, nargs=3)
    collect.add_argument("--solutes", type=int, default=1); collect.add_argument("--vacancies", type=int, default=1)
    collect.add_argument("--temperature", type=float); collect.add_argument("--seed", type=int, default=1)
    collect.add_argument("--steps", type=int, default=100); collect.add_argument("--output", required=True)
    collect.add_argument("--parameter-file"); collect.add_argument("--energy-file")
    collect.add_argument("--target", help="JSON target file; omitted uses prospective paper target")
    collect.add_argument("--reference-samples", type=int, default=0)
    collect.add_argument("--reference-max-events", type=int, default=100000)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--train", required=True); fit_parser.add_argument("--validation", required=True)
    fit_parser.add_argument("--output", required=True); fit_parser.add_argument("--hidden", type=int, default=64)
    fit_parser.add_argument("--iterations", type=int, default=300); fit_parser.add_argument("--seed", type=int, default=1)
    fit_parser.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.command == "verify":
        print(json.dumps(run_exact_smoke(args.output, args.samples), indent=2))
    elif args.command == "collect":
        adapter = make_material(args.material, size=args.size, seed=args.seed, solutes=args.solutes,
                                vacancies=args.vacancies, temperature=args.temperature,
                                parameter_file=args.parameter_file, energy_file=args.energy_file)
        if args.target:
            payload = json.loads(Path(args.target).read_text(encoding="utf-8"))
            target = Target(payload["kind"], payload["options"])
        else:
            target = paper_target(args.material, adapter)
        metadata = {"data_kind": "raw_kmc_simulation", "domain": domain_metadata(adapter, target),
                    "initial_seed": args.seed, "visitation": "passive_rate_sampled",
                    "all_legal_successors": True, "physical_policy_time_validated": False}
        rows = collect_rows(adapter, target, steps=args.steps, seed=args.seed,
                            reference_samples=args.reference_samples, reference_max_events=args.reference_max_events)
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        save_rows(output, rows, metadata)
        output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "output": str(output), "mfpt_validated": False}))
    else:
        from .network import fit
        _, report = fit(args.train, args.validation, args.output, hidden=args.hidden,
                        iterations=args.iterations, seed=args.seed, device=args.device)
        print(json.dumps(report, indent=2))
        if not report["mfpt_reference_validated"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
