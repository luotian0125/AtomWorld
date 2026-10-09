"""Three material kernels expose rates and clock-free microscopic execution.

Snapshot/restore includes caches and counters. Preview never samples a clock
or invokes an actor. Full microstate features are a correctness reference,
not a claimed scalable representation for production lattices.
"""
from copy import deepcopy
import numpy as np


class MaterialAdapter:
    def __init__(self, material, kernel):
        if material not in {"bcc", "fcc", "hcp"}:
            raise ValueError("unknown material")
        self.material, self.kernel = material, kernel

    def snapshot(self):
        return deepcopy(self.kernel.__dict__)

    def restore(self, snapshot):
        self.kernel.__dict__.clear()
        self.kernel.__dict__.update(deepcopy(snapshot))

    def rates(self):
        if self.material == "hcp":
            return self.kernel.event_catalog().rates.reshape(-1).astype(np.float64)
        return self.kernel.get_base_rates_np(copy=True).reshape(-1).astype(np.float64)

    def apply(self, event):
        rates = self.rates()
        if not isinstance(event, (int, np.integer)) or not 0 <= event < len(rates) or rates[event] <= 0:
            raise ValueError("event must be legal with positive physical rate")
        k = self.kernel
        if self.material == "hcp":
            return k.apply_event(int(event), delta_time_s=0.0, clock_owner="unassigned")
        vacancy, direction = divmod(int(event), len(k.NN1))
        info = k._apply_single_event(vacancy, direction)
        if not info["legal_event"]:
            raise RuntimeError("positive rate and transition legality disagree")
        k.energy_last += float(info["delta_e_exact"])
        k.energy_best = min(k.energy_best, k.energy_last)
        k._env_step_counter += 1
        k._refresh_after_event(info)
        return info

    def features(self):
        k = self.kernel
        if self.material == "hcp":
            # Particle identities retained; local-only pooling can alias states.
            positions = k.lattice.hydrogen_sites
            occupancy = k.lattice.occupancy
            return np.concatenate([occupancy.astype(float), positions.astype(float)])
        return np.concatenate([k.lattice.reshape(-1).astype(float), k.vacancy_pos.reshape(-1).astype(float)])

    def state_key(self):
        return self.features().tobytes()

    def energy_eV(self):
        return float(self.kernel.energy_eV if self.material == "hcp" else self.kernel.energy_last)


def make_material(material, *, size=None, seed=1, solutes=1, vacancies=1,
                  temperature=None, parameter_file=None, energy_file=None):
    """Use existing rate laws/parameters; no synthetic figure data is read."""
    size = size or ((3, 3, 3) if material == "hcp" else (2, 2, 2))
    if material == "hcp":
        from .backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
        from .backends.hcp.physics.local_energy import load_local_energy_model
        energy = load_local_energy_model(energy_file) if energy_file else None
        k = HCPHydrogenKMC(size=size, n_hydrogen=solutes, temperature_K=temperature or 600,
                           seed=seed, parameter_file=parameter_file, energy_model=energy)
    else:
        if material == "bcc":
            from .backends.bcc.config.parser import get_config
            from .backends.bcc.envs.kmc import KMCEnv
        elif material == "fcc":
            from .backends.fcc.config.parser import get_config
            from .backends.fcc.envs.kmc import KMCEnv
        else:
            raise ValueError("unknown material")
        args = get_config().parse_args([])
        args.lattice_size = list(size)
        args.lattice_v_nums = vacancies
        args.lattice_cu_nums = solutes
        args.lattice_al_nums = solutes
        args.lattice_cr_nums = 0
        args.temperature = temperature or (600 if material == "bcc" else 873)
        args.device = "cpu"
        if material == "fcc":
            args.fcc_parameter_set = "mao2012_lkmc1"
            args.fcc_parameter_file = parameter_file or ""
        # Legacy initializers use global NumPy RNG; restore it after construction.
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            k = KMCEnv(args)
            k.reset()
        finally:
            np.random.set_state(state)
    return MaterialAdapter(material, k)
