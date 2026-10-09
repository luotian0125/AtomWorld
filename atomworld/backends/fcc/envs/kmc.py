import torch
from atomworld.backends.fcc.envs.lattice import AL, CR, NI, VACANCY, Lattice
from typing import Optional, Sequence
from scipy.spatial import cKDTree
import numpy as np
import os
from atomworld.backends.fcc.config.parser import get_config
from atomworld.backends.fcc.physics.fcc_parameters import load_fcc_parameters

def _calcul_dene_v1_ppair2_vec(it_arr: np.ndarray, it2_arr: np.ndarray, nn: int, sign: float, pair_energies: list) -> np.ndarray:
    V = VACANCY
    try:
        # 获取势能矩阵 V_nn (NSPECIES x NSPECIES)
        vij_nn = pair_energies[nn - 1] 
    except IndexError:
        return np.zeros_like(it_arr, dtype=float)

    # 核心：使用 NumPy 高级索引 (Advanced Indexing)
    # it_arr 和 it2_arr 必须是相同长度的整数数组
    # 这会返回一个与 it_arr/it2_arr 形状相同的数组
    vij_new = vij_nn[it_arr, it2_arr]
    vij_old = vij_nn[V, it2_arr]
    
    # 返回 M 个修正项的数组
    return 0.5 * sign * (vij_new - vij_old)

def _calcul_dene_v2_ppair2_vec(it_arr: np.ndarray, it2_arr: np.ndarray, nn1: int, nn2: int, pair_energies: list) -> np.ndarray:
    V = VACANCY
    try:
        vij_nn1 = pair_energies[nn1 - 1] 
        vij_nn2 = pair_energies[nn2 - 1] 
    except IndexError:
        return np.zeros_like(it_arr, dtype=float)

    # 核心：使用 NumPy 高级索引 (Advanced Indexing)
    term_A_N_nn1 = vij_nn1[it_arr, it2_arr]
    term_A_N_nn2 = vij_nn2[it_arr, it2_arr]
    term_V_N_nn1 = vij_nn1[V, it2_arr] # V 是标量，it2_arr 是 M 数组，NumPy 广播
    term_V_N_nn2 = vij_nn2[V, it2_arr]
    
    # 返回 M 个修正项的数组
    return 0.5 * ( term_A_N_nn1 - term_A_N_nn2 - term_V_N_nn1 + term_V_N_nn2 )

class KMC(Lattice):
    def __init__(self, args):
        super().__init__(args)
        self.fcc_parameters = load_fcc_parameters(args)
        self.parameter_metadata = self.fcc_parameters.metadata()
        self.pair_energies = self.fcc_parameters.pair_energies.copy()
        self.E_a0 = self.fcc_parameters.migration_barriers.copy()
        self.saddle_point_energies = self.fcc_parameters.saddle_point_energies.copy()
        self.attempt_frequencies = self.fcc_parameters.attempt_frequencies.copy()
        self.rate_model = self.fcc_parameters.rate_model
        self.time = 0.0
        self.energy_history = []
        self.time_history = []
        self._plotter = None
        self.args = args

    def _get_plotter(self):
        """Import plotting dependencies only when a visualization is requested."""
        if self._plotter is None:
            from atomworld.backends.fcc.plot.fcc_plotter import FCCPlotter

            self._plotter = FCCPlotter()
        return self._plotter

    # 1. 单原子能量计算：支持任意层近邻
    def _caculate_atomic_energy(self, i, j, k, lattice=None):
        if lattice is None:
            lattice = self.lattice

        center_type = lattice[i, j, k]
        nx, ny, nz = lattice.shape
        center_pos = np.array([i, j, k])
        
        total_energy = 0.0
        
        # 定义所有近邻层
        neighbor_vectors_list = [self.NN1, self.NN2, self.NN3, self.NN4]
        
        for layer_idx, nn_vecs in enumerate(neighbor_vectors_list):
            # nn_vecs shape: (K, 3)
            nn_coords = (center_pos + nn_vecs) % [nx, ny, nz]
            
            # 获取邻居原子类型
            nn_types = lattice[nn_coords[:, 0], nn_coords[:, 1], nn_coords[:, 2]]
            
            # 查表获取能量并求和 (layer_idx 对应 pair_energies 的第一维)
            # pair_energies shape: (4, 3, 3)
            layer_energy = np.sum(self.pair_energies[layer_idx][center_type][nn_types])
            total_energy += layer_energy

        return total_energy


    # 2. 系统总能量计算：支持任意层近邻
    def calculate_system_energy(self):
        nx, ny, nz = self.lattice.shape
        coords = self.coords  # (N_sites, 3)

        center_types = self.lattice[tuple(coords.T)]  # (N_sites,)
        
        total_energy_sum = 0.0
        neighbor_vectors_list = [self.NN1, self.NN2, self.NN3, self.NN4]

        for layer_idx, nn_vecs in enumerate(neighbor_vectors_list):
            # 向量化计算所有原子的邻居
            # coords: (N, 1, 3), nn_vecs: (1, K, 3) -> nn_coords: (N, K, 3)
            nn_coords = (coords[:, None, :] + nn_vecs[None, :, :]) % [nx, ny, nz]
            
            # 获取邻居类型 (N, K)
            neighbor_types = self.lattice[tuple(nn_coords.transpose(2, 0, 1))]
            
            # 查表计算能量
            energies = self.pair_energies[layer_idx][center_types[:, None], neighbor_types]
            total_energy_sum += np.sum(energies)

        # 总能量 = sum(pair_energies) / 2
        return total_energy_sum / 2.0

    def _get_all_affected_sites(self, positions: np.ndarray) -> np.ndarray:
        """返回位置本身以及 1nn 到 4nn 的集合，并去重和模运算"""
        nx, ny, nz = self.lattice.shape
        dims = np.array([nx, ny, nz])

        # 包含 1nn 到 4nn 的所有邻居向量
        nn_vectors = np.concatenate([self.NN1, self.NN2, self.NN3, self.NN4], axis=0)
        
        # 扩展位置以包含所有邻居 (M, N_nn, 3)
        extended_pos = positions[:, None, :] + nn_vectors[None, :, :]
        
        # 合并位置本身和所有邻居，并去重
        all_sites = np.unique(
            np.concatenate([positions, extended_pos.reshape(-1, 3)], axis=0), 
            axis=0
        )
        return (all_sites % dims)

    def _batch_e_v(self, positions, lat):
        """
        计算 positions 上每个原子的对势能 E_V(r) (eatom 的 e_v 部分，支持 1NN 到 4NN)。
        """
        nx, ny, nz = self.lattice.shape
        dims = np.array([nx, ny, nz])
        
        # 转换为索引
        def get_indices(pos):
            p_mod = pos % dims
            return tuple(p_mod.T)

        center_types = lat[get_indices(positions)] # (M,)
        total_energy = np.zeros(len(center_types))

        # 遍历四个近邻层
        neighbor_vectors_list = [self.NN1, self.NN2, self.NN3, self.NN4]
        
        for layer_idx, nn_vecs in enumerate(neighbor_vectors_list):
            K = len(nn_vecs)  # FCC shell counts are (12, 6, 24, 12).
            nn_coords = (positions[:, None, :] + nn_vecs) % dims
            
            # 获取当前层所有邻居的原子类型 (M, K)
            nn_types = lat[get_indices(nn_coords.reshape(-1, 3))].reshape(len(positions), K)
            
            # 查表计算该层的能量贡献并累加
            e_layer = self.pair_energies[layer_idx][center_types[:, None], nn_types]
            total_energy += e_layer.sum(axis=1)

        return total_energy # (M,)
    
# --------------------------------------------------------------------------------
# KMC 类中的 calculate_diffusion_energy (精确翻译自 C 代码的 calcul_de)
# --------------------------------------------------------------------------------

# ... (KMC Class 和 _calcul_dene_v1/v2_ppair2 保持不变，假设它们支持 nn=3) ...

    def _get_side_nn1_correction_vec(self, vac_r_coord: np.ndarray, d_offset: np.ndarray, it_arr: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """
        计算 M 个空位同时跳跃的 6 个侧向 1NN 修正项的向量化版本。
        """
        r_V_coord = vac_r_coord
        # M 为当前批次的空位数
        M = len(vac_r_coord) 
        
        # 修正项初始化为一个 M 长度的零数组
        correction = np.zeros(M, dtype=float)
        
        dims = np.array(self.lattice.shape)
        # 确保 get_pbc_coord 函数能够处理 (M, 3) 形状的数组
        def get_pbc_coord(r_base, delta):
            # r_base: (M, 3)
            # delta: (3,) 或 (M, 3) - 这里 d_offset 是 (3,)，NumPy 会自动广播
            return (r_base + delta) % dims

        # 所有偏移量（6个）
        offsets = [
            np.array([-d_offset[0], d_offset[1], d_offset[2]]),   # C(2)
            np.array([d_offset[0], -d_offset[1], -d_offset[2]]),  # C(3)
            np.array([d_offset[0], d_offset[1], -d_offset[2]]),   # C(2')
            np.array([-d_offset[0], -d_offset[1], d_offset[2]]),  # C(3')
            np.array([d_offset[0], -d_offset[1], d_offset[2]]),   # C(2'')
            np.array([-d_offset[0], d_offset[1], -d_offset[2]])   # C(3'')
        ]
        
        # 对应的 nn1, nn2 对
        nn_pairs = [
            (1, 2), (1, 3), (1, 2), 
            (1, 3), (1, 2), (1, 3)
        ]

        # 循环遍历 6 个修正项
        for offset, (nn1, nn2) in zip(offsets, nn_pairs):
            # 1. 计算 M 个空位的 M 个邻居坐标 r_N (形状: (M, 3))
            r_N = get_pbc_coord(r_V_coord, offset)
            
            # 2. 从晶格中获取 M 个邻居的物种类型 it2 (形状: (M,))
            # 使用 r_N.T 来正确索引 3D 晶格数组
            it2_arr = lattice[tuple(r_N.T)]
            
            # 3. 调用向量化函数计算 M 个修正项
            correction += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr, nn1, nn2, self.pair_energies)

        return correction


    def _vacancy_diffusion_energy(self, i, j, k):
        """
        返回单个 vacancy 在全部 1NN 方向上的 barrier（FCC 为 12 个）。
        非法动作返回 +inf。
        与 calculate_diffusion_energy() 完全一致。
        """
        lattice = self.lattice
        nx, ny, nz = lattice.shape
        V = VACANCY

        center = np.array([i, j, k], dtype=np.int64)
        if int(lattice[i, j, k]) != V:
            raise ValueError(f"site {(i, j, k)} is not a vacancy, got type={int(lattice[i, j, k])}")

        NN1 = np.asarray(self.NN1, dtype=np.int64)
        dims = np.array([nx, ny, nz], dtype=np.int64)

        targets = (center[None, :] + NN1) % dims
        neighbor_types = lattice[targets[:, 0], targets[:, 1], targets[:, 2]]

        diffusion_energies = np.full(len(NN1), np.inf, dtype=np.float64)
        min_barrier = float(getattr(self.args, "min_barrier", 0.0))

        legal_dirs = np.where(neighbor_types != V)[0]
        for dir_idx in legal_dirs:
            tgt = targets[dir_idx]
            moving_type = int(neighbor_types[dir_idx])

            raw_barrier = self._event_migration_barrier(center, tgt, moving_type, lattice=lattice)

            diffusion_energies[dir_idx] = max(min_barrier, raw_barrier)

        return diffusion_energies

    def _exact_swap_delta_e(self, vac_pos, tgt_pos, lattice=None):
        """
        精确计算一次 vacancy <-> atom 交换导致的总能量变化:
            delta_e = E_after - E_before

        只基于当前 1NN~4NN pair potentials。
        这里的结果对你当前 pair-energy 哈密顿量是精确的，
        而且不会出现“affected 区域求和再 /2”带来的边界重复计数问题。
        """
        if lattice is None:
            lattice = self.lattice

        dims = np.array(lattice.shape, dtype=np.int64)
        vac = np.asarray(vac_pos, dtype=np.int64)
        tgt = np.asarray(tgt_pos, dtype=np.int64)

        old_v = int(lattice[tuple(vac)])
        old_t = int(lattice[tuple(tgt)])

        if old_v != VACANCY:
            raise ValueError(f"vac_pos {tuple(vac)} is not a vacancy, got type={old_v}")

        if old_t == VACANCY:
            return np.inf  # 非法动作：vacancy -> vacancy

        new_v = old_t   # target atom moves into old vacancy site
        new_t = VACANCY

        delta_e = 0.0
        neighbor_vectors_list = [self.NN1, self.NN2, self.NN3, self.NN4]

        for layer_idx, nn_vecs in enumerate(neighbor_vectors_list):
            vij = self.pair_energies[layer_idx]

            # ---- 所有涉及 vac 的 pair：统一计一次 ----
            vac_nb = (vac[None, :] + nn_vecs) % dims
            vac_nb_types_old = lattice[vac_nb[:, 0], vac_nb[:, 1], vac_nb[:, 2]]

            # 如果某个邻居正好就是 tgt，那么 after 时该邻居类型应变成 vacancy(2)
            vac_hits_tgt = np.all(vac_nb == tgt[None, :], axis=1)
            vac_nb_types_new = vac_nb_types_old.copy()
            vac_nb_types_new[vac_hits_tgt] = new_t

            delta_e += np.sum(
                vij[new_v, vac_nb_types_new] - vij[old_v, vac_nb_types_old]
            )

            # ---- 所有涉及 tgt 的 pair：但排除与 vac 的那一对，避免重复计数 ----
            tgt_nb = (tgt[None, :] + nn_vecs) % dims
            tgt_nb_types_old = lattice[tgt_nb[:, 0], tgt_nb[:, 1], tgt_nb[:, 2]]

            tgt_hits_vac = np.all(tgt_nb == vac[None, :], axis=1)
            if np.any(~tgt_hits_vac):
                delta_e += np.sum(
                    vij[new_t, tgt_nb_types_old[~tgt_hits_vac]]
                    - vij[old_t, tgt_nb_types_old[~tgt_hits_vac]]
                )

        return float(delta_e)

    def _event_migration_barrier(self, vac_pos, tgt_pos, moving_type, lattice=None):
        """Return the configured activation barrier for one atom-vacancy exchange.

        ``symmetric_kra`` is the debug kinetic-Ising form Q_X + dE/2.
        ``broken_bond_saddle`` follows the literature LKMC saddle-point model:
        E_mig = E_sp^X - E_initial, where E_initial is the sum of the
        atom-centred and vacancy-centred bonds that are broken by the exchange.
        The mutual atom-vacancy bond is intentionally present in both centred
        sums, matching the published broken-bond convention.
        """
        if lattice is None:
            lattice = self.lattice

        if self.rate_model == "symmetric_kra":
            delta_e = self._exact_swap_delta_e(vac_pos, tgt_pos, lattice=lattice)
            return float(self.E_a0[moving_type] + 0.5 * delta_e)

        if self.rate_model == "broken_bond_saddle":
            vac = tuple(np.asarray(vac_pos, dtype=np.int64))
            tgt = tuple(np.asarray(tgt_pos, dtype=np.int64))
            initial_bond_energy = self._caculate_atomic_energy(*vac, lattice=lattice)
            initial_bond_energy += self._caculate_atomic_energy(*tgt, lattice=lattice)
            return float(self.saddle_point_energies[moving_type] - initial_bond_energy)

        raise RuntimeError(f"Unsupported FCC rate model: {self.rate_model!r}")

    def _attempt_frequency_for_species(self, moving_type):
        override = getattr(self.args, "attempt_frequency", None)
        if override is not None:
            return float(override)
        return float(self.attempt_frequencies[moving_type])

    def _attempt_frequency_matrix_for_indices(self, vacancy_indices):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        attempts = np.zeros((len(vacancy_indices), len(self.NN1)), dtype=np.float64)
        if vacancy_indices.size == 0:
            return attempts

        override = getattr(self.args, "attempt_frequency", None)
        if override is not None:
            attempts.fill(float(override))
            return attempts

        dims = np.asarray(self.lattice.shape, dtype=np.int64)
        vacancies = np.asarray(self.vacancy_pos, dtype=np.int64)[vacancy_indices]
        targets = (vacancies[:, None, :] + np.asarray(self.NN1, dtype=np.int64)[None, :, :]) % dims
        neighbor_types = self.lattice[tuple(targets.transpose(2, 0, 1))]
        legal = neighbor_types != VACANCY
        attempts[legal] = self.attempt_frequencies[neighbor_types[legal]]
        return attempts


    def calculate_diffusion_energy(self):
        """
        返回 shape = (N_vac, N_1NN) 的迁移势垒矩阵。
        非法动作直接置为 +inf，保证后续 rate=0。
        """
        lattice = self.lattice
        nx, ny, nz = lattice.shape
        V = VACANCY

        vacancies = np.asarray(self.vacancy_pos, dtype=np.int64)
        NN1 = np.asarray(self.NN1, dtype=np.int64)
        dims = np.array([nx, ny, nz], dtype=np.int64)

        diffusion_energies = np.full((len(vacancies), len(NN1)), np.inf, dtype=np.float64)
        min_barrier = float(getattr(self.args, "min_barrier", 0.0))  # 建议默认 0.0

        for vac_idx, vac in enumerate(vacancies):
            targets = (vac[None, :] + NN1) % dims
            neighbor_types = lattice[targets[:, 0], targets[:, 1], targets[:, 2]]

            legal_dirs = np.where(neighbor_types != V)[0]
            for dir_idx in legal_dirs:
                tgt = targets[dir_idx]
                moving_type = int(neighbor_types[dir_idx])

                raw_barrier = self._event_migration_barrier(vac, tgt, moving_type, lattice=lattice)

                # 物理上建议不允许负 barrier，避免 rate > nu
                diffusion_energies[vac_idx, dir_idx] = max(min_barrier, raw_barrier)

        return diffusion_energies

    def calculate_diffusion_rate(self):
        """
        返回 shape = (N_vac, N_1NN) 的速率矩阵。
        非法动作(+inf barrier) -> rate = 0。
        """
        kB = 8.617333262145e-5  # eV/K
        T = float(self.args.temperature)
        barriers = np.asarray(self.calculate_diffusion_energy(), dtype=np.float64)
        attempts = self._attempt_frequency_matrix_for_indices(np.arange(len(self.vacancy_pos)))

        # 非法动作保持 +inf；合法动作 barrier 至少为 0
        barriers = np.where(np.isfinite(barriers), np.maximum(barriers, 0.0), np.inf)

        with np.errstate(over='ignore', under='ignore'):
            rates = attempts * np.exp(-barriers / (kB * T))

        rates[~np.isfinite(barriers)] = 0.0
        rates = np.minimum(np.maximum(rates, 0.0), attempts)

        return rates


    def stastic_local_atoms(self, i: int, j: int, k: int) -> np.ndarray:
        nx, ny, nz = self.lattice.shape
        local_env = np.zeros(8, dtype=int)
        for offset, NN in zip([0, 4], [self.NN1, self.NN2]):
            coords = (np.array([i, j, k]) + NN) % [nx, ny, nz]
            types = self.lattice[tuple(coords.T)]
            counts = np.bincount(types, minlength=4)[:4]
            local_env[offset:offset + 4] = counts
        return local_env

    def static_vancancy_local_atoms(self) -> np.ndarray:
            """
            向量化构造每个 vacancy 的 [1NN + 2NN + 3NN + 4NN] 原子种类特征。
            输出 shape: (n_vacancies, 54)
            其中 54 = 12(1NN) + 6(2NN) + 24(3NN) + 12(4NN)
            """
            lattice = self.lattice
            nx, ny, nz = lattice.shape
            vacancy_pos = np.array(self.vacancy_pos)  # shape: (N, 3)

            # 收集所有邻居向量
            # 注意：顺序很重要，必须固定！建议按距离远近拼接
            # 1NN(12) + 2NN(6) + 3NN(24) + 4NN(12)
            neighbor_vectors = [
                self.NN1, 
                self.NN2, 
                self.NN3, 
                self.NN4
            ]
            
            all_neighbor_types = []

            for NN_k in neighbor_vectors:
                # NN_k shape: [K, 3]
                # 计算邻居坐标: (N, K, 3)
                nn_coords = (vacancy_pos[:, None, :] + NN_k[None, :, :]) % [nx, ny, nz]
                
                # 查原子类型: (N, K)
                nn_types = lattice[nn_coords[:, :, 0], nn_coords[:, :, 1], nn_coords[:, :, 2]]
                all_neighbor_types.append(nn_types)

            # 拼接所有层的特征
            # result shape: (N, 12+6+24+12) = (N, 54)
            return np.concatenate(all_neighbor_types, axis=1).astype(np.int32)

    def step(self):
        rates = self.calculate_diffusion_rate()
        flat_rates = []
        vac_indices = []
        dir_indices = []

        for vac_idx, vac_rates in enumerate(rates):
            for dir_idx, rate in enumerate(vac_rates):
                if rate > 0:
                    flat_rates.append(rate)
                    vac_indices.append(vac_idx)
                    dir_indices.append(dir_idx)

        if not flat_rates:
            return 0.0

        total_rate = np.sum(flat_rates)
        r = np.random.rand() * total_rate
        chosen_idx = np.searchsorted(np.cumsum(flat_rates), r)

        vac_idx = vac_indices[chosen_idx]
        dir_idx = dir_indices[chosen_idx]
        vi, vj, vk = self.vacancy_pos[vac_idx]
        nn = self.NN1[dir_idx]
        ai = (vi + nn[0]) % self.lattice.shape[0]
        aj = (vj + nn[1]) % self.lattice.shape[1]
        ak = (vk + nn[2]) % self.lattice.shape[2]

        moving_type = int(self.lattice[ai, aj, ak])
        self.lattice[vi, vj, vk], self.lattice[ai, aj, ak] = self.lattice[ai, aj, ak], self.lattice[vi, vj, vk]
        self.vacancy_pos[vac_idx] = (ai, aj, ak)
        if moving_type in (AL, CR):
            self.update_cu_pos()
        delta_t = -np.log(max(np.random.rand(), np.finfo(float).tiny)) / total_rate
        self.time += delta_t
        self.energy_history.append(self.calculate_system_energy())
        self.time_history.append(self.time)
        return delta_t

    @staticmethod
    def _prepare_save_dir(save_path: Optional[str]) -> None:
        if not save_path:
            return
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def visualize(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_vacancy: bool = False,
        constant: float = 1.0,
    ):
        self._prepare_save_dir(save_path)
        return self._get_plotter().plot_lattice(
            self.lattice, 
            constant,
            title=title,
            save_path=save_path,
            show_vacancy=show_vacancy,
        )

    def visualize_snapshots(
        self,
        lattices: Sequence[np.ndarray],
        titles: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        show_vacancy: bool = False,
        constant: float = 1.0,
        suptitle: Optional[str] = None,
    ):
        self._prepare_save_dir(save_path)
        return self._get_plotter().plot_lattice_snapshots(
            lattices=lattices,
            constant=constant,
            titles=titles,
            save_path=save_path,
            show_vacancy=show_vacancy,
            suptitle=suptitle,
        )

    def collect_lattice_snapshots(self, snapshot_steps: Sequence[int]):
        """Advance the current KMC state and collect lattice copies at selected event counts."""
        if not snapshot_steps:
            raise ValueError("snapshot_steps must contain at least one step index")

        targets = sorted({int(step) for step in snapshot_steps})
        if targets[0] < 0:
            raise ValueError("snapshot_steps must be non-negative")

        lattices = []
        physical_times = []
        current_step = 0
        target_idx = 0

        while target_idx < len(targets) and targets[target_idx] == 0:
            lattices.append(self.lattice.copy())
            physical_times.append(float(self.time))
            target_idx += 1

        while target_idx < len(targets):
            delta_t = self.step()
            if delta_t == 0.0:
                raise RuntimeError(
                    f"simulation became inactive before reaching snapshot step {targets[target_idx]}"
                )
            current_step += 1
            while target_idx < len(targets) and current_step >= targets[target_idx]:
                lattices.append(self.lattice.copy())
                physical_times.append(float(self.time))
                target_idx += 1

        return lattices, physical_times

    def simulate_snapshot_panel(
        self,
        snapshot_steps: Sequence[int],
        save_path: Optional[str] = None,
        titles: Optional[Sequence[str]] = None,
        show_vacancy: bool = False,
        constant: float = 1.0,
        suptitle: Optional[str] = None,
    ):
        """Run KMC up to the requested event counts and save a multi-snapshot panel."""
        lattices, physical_times = self.collect_lattice_snapshots(snapshot_steps)
        ordered_steps = sorted({int(step) for step in snapshot_steps})

        if titles is None:
            titles = [f"step={step}" for step in ordered_steps]

        fig, axes = self.visualize_snapshots(
            lattices=lattices,
            titles=titles,
            save_path=save_path,
            show_vacancy=show_vacancy,
            constant=constant,
            suptitle=suptitle,
        )
        return fig, axes, physical_times
    
    def plot_energy_history(self, save_path: Optional[str] = None):
        self._prepare_save_dir(save_path)
        return self._get_plotter().plot_energy_evolution(
            self.energy_history,
            self.time_history,
            save_path=save_path
        )


class KMCEnv(KMC):
    """
    Stage-1 single-event RL environment built on top of the KMC physics kernel.

    Key change from the previous multi-agent batch-execution version:
    each env step now executes exactly one global event
        event_id = vac_idx * N_1NN + dir_idx
    while preserving vacancy-wise local observations for the actor encoder.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_agents = args.lattice_v_nums
        self.energy_last = self.calculate_system_energy()
        self.max_episode_steps = int(args.num_env_steps) // args.episode_length
        self.history_len = int(getattr(args, "history_len", 5))
        self.sparse_defect_max_tokens = int(getattr(args, "sparse_defect_max_tokens", 24))
        self.sparse_defect_shell_cutoff = int(getattr(args, "sparse_defect_shell_cutoff", 16))
        self.sparse_defect_feature_dim = 6
        self.event_feature_dim = max(9, int(getattr(args, "event_feature_dim", 9)))
        self.episode_length = int(getattr(args, "episode_length", 1))
        self.enable_energy_check = bool(getattr(args, "enable_energy_check", False))
        configured_energy_check_interval = int(getattr(args, "energy_check_interval", 200))
        self.energy_check_interval = max(0, configured_energy_check_interval) if self.enable_energy_check else 0
        self.energy_check_tol = float(getattr(args, "energy_check_tol", 1e-5))
        self.system_stats_interval = max(1, int(getattr(args, "system_stats_interval", 1)))
        self.skip_policy_context = bool(getattr(args, "skip_policy_context", False))
        self._env_step_counter = 0
        self.energy_best = None

        self._cached_system_stats = None
        self._last_system_stats_step = -10**9
        self._cached_base_rates = None
        self._cached_event_features = None
        self._cached_sparse_tokens = None
        self._cached_sparse_masks = None
        self._cached_local_spatial_obs = None

        self.action_history_array = np.full((self.num_agents, self.history_len), -1, dtype=np.int32)

    def _compute_system_stats(self):
        lattice_size = np.asarray(self.lattice.shape, dtype=np.float64)
        a = 2.0
        cu_pos = np.asarray(self.cu_pos) % lattice_size if len(self.cu_pos) > 0 else np.empty((0, 3), dtype=np.float64)
        vac_pos = np.asarray(self.vacancy_pos) % lattice_size if len(self.vacancy_pos) > 0 else np.empty((0, 3), dtype=np.float64)
        stats = []

        if cu_pos.size > 0:
            cu_mean = cu_pos.mean(axis=0)
            cu_std = cu_pos.std(axis=0)
        else:
            cu_mean = np.zeros(3, dtype=np.float64)
            cu_std = np.zeros(3, dtype=np.float64)
        stats.extend([*cu_mean, *cu_std])

        cu_cv = 0.0
        cu_tree = cKDTree(cu_pos, boxsize=lattice_size) if len(cu_pos) >= 2 else None
        if cu_tree is not None:
            counts = np.asarray(cu_tree.query_ball_point(cu_pos, r=2 * a, return_length=True)) - 1
            mean_counts = float(counts.mean())
            if mean_counts > 0.0:
                cu_cv = float(counts.std() / mean_counts)
        stats.append(cu_cv)

        avg_vac_to_cu = 0.0
        if cu_tree is not None and vac_pos.size > 0:
            dists, _ = cu_tree.query(vac_pos, k=1)
            avg_vac_to_cu = float(np.mean(dists))
        stats.append(avg_vac_to_cu)

        cu_near_ratio = 0.0
        if cu_pos.size > 0 and vac_pos.size > 0:
            vac_tree = cKDTree(vac_pos, boxsize=lattice_size)
            nearby_counts = np.asarray(vac_tree.query_ball_point(cu_pos, r=2 * a, return_length=True))
            cu_near_ratio = float((nearby_counts > 0).mean())
        stats.append(cu_near_ratio)

        return np.asarray(stats, dtype=np.float32)

    def get_system_stats(self, force=False):
        if force or self._cached_system_stats is None or (self._env_step_counter - self._last_system_stats_step) >= self.system_stats_interval:
            self._cached_system_stats = self._compute_system_stats()
            self._last_system_stats_step = int(self._env_step_counter)
        return self._cached_system_stats.copy()

    def _invalidate_state_caches(self, invalidate_stats=False):
        self._cached_base_rates = None
        self._cached_event_features = None
        self._cached_sparse_tokens = None
        self._cached_sparse_masks = None
        self._cached_local_spatial_obs = None
        if invalidate_stats:
            self._cached_system_stats = None
            self._last_system_stats_step = -10**9

    def _refresh_all_caches(self, force_stats=False):
        self._cached_base_rates = np.asarray(self.calculate_diffusion_rate(), dtype=np.float32)
        full_indices = np.arange(len(self.vacancy_pos), dtype=np.int64)
        if self.skip_policy_context:
            self._cached_event_features = self._empty_event_features(len(full_indices))
            self._cached_event_features[..., 0] = (self._cached_base_rates > 0.0).astype(np.float32)
            self._cached_sparse_tokens = np.zeros(
                (len(full_indices), self.sparse_defect_max_tokens, self.sparse_defect_feature_dim),
                dtype=np.float32,
            )
            self._cached_sparse_tokens[..., 0] = 2.0
            self._cached_sparse_masks = np.zeros(
                (len(full_indices), self.sparse_defect_max_tokens), dtype=np.float32
            )
            self._cached_local_spatial_obs = np.zeros(
                (len(full_indices), self.spatial_obs_dim), dtype=np.float32
            )
        else:
            self._cached_event_features = self._calculate_event_features_for_indices(
                full_indices, base_rates=self._cached_base_rates
            )
            self._cached_sparse_tokens, self._cached_sparse_masks = self._compute_sparse_context_for_indices(full_indices)
            self._cached_local_spatial_obs = self._compute_local_atoms_for_indices(full_indices)
        if force_stats or self._cached_system_stats is None:
            self._cached_system_stats = self._compute_system_stats()
            self._last_system_stats_step = int(self._env_step_counter)

    def _minimal_image_offsets(self, ref_pos, target_pos):
        dims = np.asarray(self.lattice.shape, dtype=np.int32)
        ref_pos = np.asarray(ref_pos, dtype=np.int32)
        target_pos = np.asarray(target_pos, dtype=np.int32)
        delta = target_pos - ref_pos
        half = dims // 2
        delta = np.where(delta > half, delta - dims, delta)
        delta = np.where(delta < -half, delta + dims, delta)
        return delta.astype(np.int32)

    def _minimal_image_offsets_broadcast(self, ref_pos, target_pos):
        dims = np.asarray(self.lattice.shape, dtype=np.int32)
        ref_pos = np.asarray(ref_pos, dtype=np.int32)
        target_pos = np.asarray(target_pos, dtype=np.int32)
        delta = target_pos - ref_pos
        half = dims // 2
        delta = np.where(delta > half, delta - dims, delta)
        delta = np.where(delta < -half, delta + dims, delta)
        return delta.astype(np.int32)

    @staticmethod
    def _row_view(arr: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(arr)
        return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1]))).reshape(-1)

    def _compute_local_atoms_for_indices(self, vacancy_indices):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        if vacancy_indices.size == 0:
            return np.zeros((0, self.spatial_obs_dim), dtype=np.float32)

        lattice = self.lattice
        nx, ny, nz = lattice.shape
        vacancy_pos = np.asarray(self.vacancy_pos, dtype=np.int64)[vacancy_indices]
        neighbor_vectors = [self.NN1, self.NN2, self.NN3, self.NN4]
        all_neighbor_types = []

        for nn_k in neighbor_vectors:
            nn_coords = (vacancy_pos[:, None, :] + nn_k[None, :, :]) % [nx, ny, nz]
            nn_types = lattice[nn_coords[:, :, 0], nn_coords[:, :, 1], nn_coords[:, :, 2]]
            all_neighbor_types.append(nn_types)

        return np.concatenate(all_neighbor_types, axis=1).astype(np.float32, copy=False)

    def _compute_sparse_context_for_indices(self, vacancy_indices):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        max_tokens = int(self.sparse_defect_max_tokens)
        feat_dim = int(self.sparse_defect_feature_dim)
        n_rows = int(vacancy_indices.size)

        tokens = np.zeros((n_rows, max_tokens, feat_dim), dtype=np.float32)
        tokens[..., 0] = 2.0
        masks = np.zeros((n_rows, max_tokens), dtype=np.float32)

        if n_rows == 0:
            return tokens, masks

        cu_pos = np.asarray(self.cu_pos, dtype=np.int32) if len(self.cu_pos) > 0 else np.empty((0, 3), dtype=np.int32)
        vac_pos = np.asarray(self.vacancy_pos, dtype=np.int32) if len(self.vacancy_pos) > 0 else np.empty((0, 3), dtype=np.int32)
        if vac_pos.shape[0] == 0:
            return tokens, masks

        ref_vac = vac_pos[vacancy_indices]
        offset_parts = []
        type_parts = []
        dist2_parts = []
        valid_parts = []

        if cu_pos.shape[0] > 0:
            cu_offsets = self._minimal_image_offsets_broadcast(ref_vac[:, None, :], cu_pos[None, :, :])
            cu_dist2 = np.sum(cu_offsets.astype(np.int64) ** 2, axis=-1)
            offset_parts.append(cu_offsets)
            type_parts.append(np.zeros(cu_dist2.shape, dtype=np.int32))
            dist2_parts.append(cu_dist2)
            valid_parts.append(np.ones(cu_dist2.shape, dtype=bool))

        if vac_pos.shape[0] > 0:
            all_vac_offsets = self._minimal_image_offsets_broadcast(ref_vac[:, None, :], vac_pos[None, :, :])
            all_vac_dist2 = np.sum(all_vac_offsets.astype(np.int64) ** 2, axis=-1)
            vac_valid = np.ones(all_vac_dist2.shape, dtype=bool)
            vac_valid[np.arange(n_rows), vacancy_indices] = False
            all_vac_dist2 = np.where(vac_valid, all_vac_dist2, np.iinfo(np.int64).max)
            offset_parts.append(all_vac_offsets)
            type_parts.append(np.ones(all_vac_dist2.shape, dtype=np.int32))
            dist2_parts.append(all_vac_dist2)
            valid_parts.append(vac_valid)

        if not offset_parts:
            return tokens, masks

        offsets = np.concatenate(offset_parts, axis=1)
        defect_types = np.concatenate(type_parts, axis=1)
        dist2 = np.concatenate(dist2_parts, axis=1)
        valid = np.concatenate(valid_parts, axis=1)

        # Only the closest max_tokens defects can reach the output. Selecting
        # that subset first avoids sorting every solute after every event.
        candidate_count = min(max_tokens, dist2.shape[1])
        if candidate_count <= 0:
            return tokens, masks
        if candidate_count < dist2.shape[1]:
            thresholds = np.partition(dist2, kth=candidate_count - 1, axis=1)[:, candidate_count - 1]
            order = np.empty((dist2.shape[0], candidate_count), dtype=np.int64)
            for row_index, threshold in enumerate(thresholds):
                nearer = np.flatnonzero(dist2[row_index] < threshold)
                tied = np.flatnonzero(dist2[row_index] == threshold)
                needed = candidate_count - len(nearer)
                selected = np.concatenate([nearer, tied[:needed]])
                local_order = np.argsort(dist2[row_index, selected], kind="stable")
                order[row_index] = selected[local_order]
        else:
            order = np.argsort(dist2, axis=1, kind="stable")
        sorted_offsets = np.take_along_axis(offsets, order[..., None], axis=1)
        sorted_types = np.take_along_axis(defect_types, order, axis=1)
        sorted_dist2 = np.take_along_axis(dist2, order, axis=1)
        sorted_valid = np.take_along_axis(valid, order, axis=1)

        shell_rank = np.ones_like(sorted_dist2, dtype=np.int32)
        if sorted_dist2.shape[1] > 1:
            shell_rank[:, 1:] = 1 + np.cumsum(sorted_dist2[:, 1:] != sorted_dist2[:, :-1], axis=1)

        keep = sorted_valid & (shell_rank <= int(self.sparse_defect_shell_cutoff))
        selected_rank = np.cumsum(keep, axis=1) - 1
        select = keep & (selected_rank < max_tokens)

        rows, cols = np.nonzero(select)
        if rows.size > 0:
            dst = selected_rank[rows, cols]
            tokens[rows, dst, 0] = sorted_types[rows, cols].astype(np.float32)
            tokens[rows, dst, 1:4] = sorted_offsets[rows, cols].astype(np.float32)
            tokens[rows, dst, 4] = sorted_dist2[rows, cols].astype(np.float32)
            tokens[rows, dst, 5] = shell_rank[rows, cols].astype(np.float32)
            masks[rows, dst] = 1.0

        return tokens, masks

    def get_sparse_defect_context(self, force=False, copy=True):
        if force or self._cached_sparse_tokens is None or self._cached_sparse_masks is None:
            full_indices = np.arange(len(self.vacancy_pos), dtype=np.int64)
            self._cached_sparse_tokens, self._cached_sparse_masks = self._compute_sparse_context_for_indices(full_indices)
        if copy:
            return self._cached_sparse_tokens.copy(), self._cached_sparse_masks.copy()
        return self._cached_sparse_tokens, self._cached_sparse_masks

    def _empty_event_features(self, n_rows):
        return np.zeros((int(n_rows), len(self.NN1), self.event_feature_dim), dtype=np.float32)

    def _calculate_event_features_for_indices(self, vacancy_indices, base_rates=None):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        if vacancy_indices.size == 0:
            return self._empty_event_features(0)

        lattice = self.lattice
        dims = np.array(lattice.shape, dtype=np.int64)
        vacancies = np.asarray(self.vacancy_pos, dtype=np.int64)[vacancy_indices]
        NN1 = np.asarray(self.NN1, dtype=np.int64)
        features = self._empty_event_features(len(vacancy_indices))

        kB = 8.617333262145e-5
        T = float(self.args.temperature)
        min_barrier = float(getattr(self.args, "min_barrier", 0.0))

        if base_rates is not None:
            base_rates = np.asarray(base_rates, dtype=np.float64).reshape(len(vacancy_indices), len(NN1))

        for row_idx, vac in enumerate(vacancies):
            targets = (vac[None, :] + NN1) % dims
            neighbor_types = lattice[targets[:, 0], targets[:, 1], targets[:, 2]]
            legal_dirs = np.where(neighbor_types != VACANCY)[0]
            for dir_idx in legal_dirs:
                tgt = targets[dir_idx]
                moving_type = int(neighbor_types[dir_idx])
                delta_e = float(self._exact_swap_delta_e(vac, tgt, lattice=lattice))
                barrier = max(
                    min_barrier,
                    self._event_migration_barrier(vac, tgt, moving_type, lattice=lattice),
                )
                nu = self._attempt_frequency_for_species(moving_type)
                if base_rates is None:
                    rate = float(nu * np.exp(-barrier / (kB * T)))
                    rate = float(np.clip(rate, 0.0, nu))
                else:
                    rate = float(base_rates[row_idx, dir_idx])
                log_rate = float(np.log(max(rate, 1e-12)))

                features[row_idx, dir_idx, 0] = 1.0
                features[row_idx, dir_idx, 1] = 1.0 if moving_type == NI else 0.0
                features[row_idx, dir_idx, 2] = 1.0 if moving_type == AL else 0.0
                features[row_idx, dir_idx, 3] = 1.0 if moving_type == CR else 0.0
                features[row_idx, dir_idx, 4] = -delta_e
                features[row_idx, dir_idx, 5] = barrier
                features[row_idx, dir_idx, 6] = log_rate
                features[row_idx, dir_idx, 7] = 1.0 if delta_e < 0.0 else 0.0
                features[row_idx, dir_idx, 8] = 1.0 if delta_e > 0.0 else 0.0

        return features

    def get_event_features(self, force=False, copy=True):
        if force or self._cached_event_features is None:
            full_indices = np.arange(len(self.vacancy_pos), dtype=np.int64)
            self._cached_event_features = self._calculate_event_features_for_indices(
                full_indices, base_rates=self._get_cached_base_rates_np(force=False)
            )
        return self._cached_event_features.copy() if copy else self._cached_event_features

    def _calculate_diffusion_energy_for_indices(self, vacancy_indices):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        lattice = self.lattice
        nx, ny, nz = lattice.shape
        V = VACANCY

        vacancies = np.asarray(self.vacancy_pos, dtype=np.int64)[vacancy_indices]
        NN1 = np.asarray(self.NN1, dtype=np.int64)
        dims = np.array([nx, ny, nz], dtype=np.int64)

        diffusion_energies = np.full((len(vacancies), len(NN1)), np.inf, dtype=np.float64)
        min_barrier = float(getattr(self.args, "min_barrier", 0.0))

        for row_idx, vac in enumerate(vacancies):
            targets = (vac[None, :] + NN1) % dims
            neighbor_types = lattice[targets[:, 0], targets[:, 1], targets[:, 2]]

            legal_dirs = np.where(neighbor_types != V)[0]
            for dir_idx in legal_dirs:
                tgt = targets[dir_idx]
                moving_type = int(neighbor_types[dir_idx])
                raw_barrier = self._event_migration_barrier(vac, tgt, moving_type, lattice=lattice)
                diffusion_energies[row_idx, dir_idx] = max(min_barrier, raw_barrier)

        return diffusion_energies

    def _calculate_diffusion_rate_for_indices(self, vacancy_indices):
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int64).reshape(-1)
        if vacancy_indices.size == 0:
            return np.zeros((0, len(self.NN1)), dtype=np.float32)

        kB = 8.617333262145e-5
        T = float(self.args.temperature)
        barriers = np.asarray(self._calculate_diffusion_energy_for_indices(vacancy_indices), dtype=np.float64)
        attempts = self._attempt_frequency_matrix_for_indices(vacancy_indices)
        barriers = np.where(np.isfinite(barriers), np.maximum(barriers, 0.0), np.inf)

        with np.errstate(over='ignore', under='ignore'):
            rates = attempts * np.exp(-barriers / (kB * T))

        rates[~np.isfinite(barriers)] = 0.0
        rates = np.minimum(np.maximum(rates, 0.0), attempts)
        return rates.astype(np.float32)

    def _get_cached_base_rates_np(self, force=False):
        if force or self._cached_base_rates is None:
            self._cached_base_rates = np.asarray(self.calculate_diffusion_rate(), dtype=np.float32)
        return self._cached_base_rates

    def get_base_rates_np(self, force=False, copy=True):
        base_rates = self._get_cached_base_rates_np(force=force)
        return base_rates.copy() if copy else base_rates

    def _get_conservative_affected_vacancy_indices(self, changed_positions):
        changed_positions = np.asarray(changed_positions, dtype=np.int32).reshape(-1, 3)
        if changed_positions.size == 0 or len(self.vacancy_pos) == 0:
            return np.zeros((0,), dtype=np.int64)

        dims = np.asarray(self.lattice.shape, dtype=np.int32)
        changed_positions = changed_positions % dims
        affected_sites = np.asarray(self._get_all_affected_sites(changed_positions), dtype=np.int32) % dims

        vac_pos = np.asarray(self.vacancy_pos, dtype=np.int32) % dims
        vac_targets = (vac_pos[:, None, :] + np.asarray(self.NN1, dtype=np.int32)[None, :, :]) % dims

        affected_view = self._row_view(affected_sites)
        vac_mask = np.isin(self._row_view(vac_pos), affected_view)
        tgt_mask = np.isin(self._row_view(vac_targets.reshape(-1, 3)), affected_view).reshape(len(vac_pos), len(self.NN1)).any(axis=1)
        affected = np.flatnonzero(vac_mask | tgt_mask).astype(np.int64)

        if affected.size == 0:
            return np.arange(len(vac_pos), dtype=np.int64)
        return affected

    def _refresh_after_event(self, event_info):
        if (
            self._cached_base_rates is None
            or self._cached_event_features is None
            or self._cached_sparse_tokens is None
            or self._cached_sparse_masks is None
            or self._cached_local_spatial_obs is None
        ):
            self._refresh_all_caches(force_stats=False)
            return

        if not bool(event_info.get("legal_event", False)):
            return

        changed_positions = np.asarray([event_info["old_pos"], event_info["new_pos"]], dtype=np.int32)
        affected_indices = self._get_conservative_affected_vacancy_indices(changed_positions)

        if affected_indices.size == 0 or affected_indices.size >= len(self.vacancy_pos):
            self._refresh_all_caches(force_stats=False)
            return

        self._cached_base_rates[affected_indices] = self._calculate_diffusion_rate_for_indices(affected_indices)
        self._cached_event_features[affected_indices] = self._calculate_event_features_for_indices(
            affected_indices, base_rates=self._cached_base_rates[affected_indices]
        )
        local_tokens, local_masks = self._compute_sparse_context_for_indices(affected_indices)
        self._cached_sparse_tokens[affected_indices] = local_tokens
        self._cached_sparse_masks[affected_indices] = local_masks
        self._cached_local_spatial_obs[affected_indices] = self._compute_local_atoms_for_indices(affected_indices)

    def get_combined_obs(self):
        if self._cached_local_spatial_obs is None:
            full_indices = np.arange(len(self.vacancy_pos), dtype=np.int64)
            self._cached_local_spatial_obs = self._compute_local_atoms_for_indices(full_indices)
        history_obs = self.action_history_array.astype(np.float32, copy=False)
        return np.concatenate([self._cached_local_spatial_obs, history_obs], axis=-1).astype(np.float32, copy=False)

    def _build_full_obs(self, obs, share_stats):
        full_obs_single = np.concatenate([obs.flatten(), share_stats], axis=0).astype(np.float32)
        return np.stack([full_obs_single] * self.num_agents, axis=0)

    def reset(self):
        self.time = 0.0
        self.energy_history = []
        self.time_history = []
        self._env_step_counter = 0
        self.init_lattice()

        initial_energy = self.calculate_system_energy()
        self.energy_last = initial_energy
        self.energy_best = float(initial_energy)
        self.energy_history.append(initial_energy)
        self.time_history.append(0.0)

        self.action_history_array = np.full((self.num_agents, self.history_len), -1, dtype=np.int32)

        self._invalidate_state_caches(invalidate_stats=True)
        self._refresh_all_caches(force_stats=True)

        initial_obs = self.get_combined_obs().astype(np.float32)
        share_stats = self.get_system_stats(force=False).astype(np.float32)
        full_obs = self._build_full_obs(initial_obs, share_stats)
        return initial_obs, full_obs

    def get_state_dict(self):
        return {
            "schema_version": 2,
            "material_system": "fcc_ni_al_cr",
            "parameter_metadata": dict(self.parameter_metadata),
            "lattice": np.asarray(self.lattice, dtype=np.int32).copy(),
            "vacancy_pos": np.asarray(self.vacancy_pos, dtype=np.int32).copy(),
            "solute_pos": np.asarray(self.solute_pos, dtype=np.int32).copy(),
            "al_pos": np.asarray(self.al_pos, dtype=np.int32).copy(),
            "cr_pos": np.asarray(self.cr_pos, dtype=np.int32).copy(),
            "cu_pos": np.asarray(self.cu_pos, dtype=np.int32).copy(),
            "time": float(self.time),
            "energy_last": float(self.energy_last),
            "energy_history": [float(x) for x in self.energy_history],
            "time_history": [float(x) for x in self.time_history],
            "action_histories": self.action_history_array.astype(np.int32).tolist(),
        }

    def set_state_dict(self, state):
        self.lattice = np.asarray(state["lattice"], dtype=np.int32).copy()
        self.vacancy_pos = np.asarray(state["vacancy_pos"], dtype=np.int32).copy()
        solute_state = state.get("solute_pos", state.get("cu_pos", []))
        self.cu_pos = np.asarray(solute_state, dtype=np.int32).reshape(-1, 3).copy()
        self._build_cu_pos_index()
        self.time = float(state["time"])
        self.energy_last = float(state["energy_last"])
        self.energy_history = [float(x) for x in state.get("energy_history", [self.energy_last])]
        self.time_history = [float(x) for x in state.get("time_history", [self.time])]
        self._env_step_counter = max(0, len(self.energy_history) - 1)
        self.energy_best = float(min(self.energy_history)) if self.energy_history else float(self.energy_last)

        restored_histories = state.get("action_histories")
        if restored_histories is None:
            restored_histories = np.full((self.num_agents, self.history_len), -1, dtype=np.int32)
        else:
            restored_histories = np.asarray(restored_histories, dtype=np.int32)
            fixed_histories = np.full((self.num_agents, self.history_len), -1, dtype=np.int32)
            if restored_histories.ndim == 1:
                restored_histories = restored_histories.reshape(1, -1)
            rows = min(self.num_agents, restored_histories.shape[0])
            cols = min(self.history_len, restored_histories.shape[1])
            fixed_histories[:rows, self.history_len - cols :] = restored_histories[:rows, restored_histories.shape[1] - cols :]
            restored_histories = fixed_histories
        self.action_history_array = restored_histories

        self._invalidate_state_caches(invalidate_stats=True)
        self._refresh_all_caches(force_stats=True)

    def _decode_event(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).reshape(-1)
        event_id = int(action[0]) if action.size > 0 else 0
        event_id = int(np.clip(event_id, 0, self.num_agents * len(self.NN1) - 1))
        vac_idx = event_id // len(self.NN1)
        dir_idx = event_id % len(self.NN1)
        return event_id, vac_idx, dir_idx

    def _apply_single_event(self, vac_idx, dir_idx):
        vi, vj, vk = map(int, self.vacancy_pos[vac_idx])
        di, dj, dk = map(int, self.NN1[dir_idx])
        nx, ny, nz = self.lattice.shape
        ni = (vi + di) % nx
        nj = (vj + dj) % ny
        nk = (vk + dk) % nz
        moving_type = int(self.lattice[ni, nj, nk])
        legal_event = moving_type != VACANCY

        event_info = {
            "selected_vacancy": vac_idx,
            "selected_direction": dir_idx,
            "old_pos": (vi, vj, vk),
            "new_pos": (ni, nj, nk),
            "moving_type": moving_type,
            "legal_event": legal_event,
            "delta_e_exact": 0.0,
            "local_energy_change": 0.0,
        }

        if not legal_event:
            return event_info

        delta_e = self._exact_swap_delta_e((vi, vj, vk), (ni, nj, nk), lattice=self.lattice)
        event_info["delta_e_exact"] = float(delta_e)
        event_info["local_energy_change"] = -float(delta_e)

        self.lattice[vi, vj, vk], self.lattice[ni, nj, nk] = moving_type, VACANCY
        self.vacancy_pos[vac_idx] = (ni, nj, nk)

        if moving_type in (AL, CR):
            solute_old_pos = (ni, nj, nk)
            solute_new_pos = (vi, vj, vk)
            if solute_old_pos in self.solute_pos_index:
                idx = self.solute_pos_index.pop(solute_old_pos)
                self.solute_pos[idx] = solute_new_pos
                self.solute_pos_index[solute_new_pos] = idx
                self.cu_pos = self.solute_pos
                self.cu_pos_index = self.solute_pos_index

        return event_info

    def step(self, action, episode=None):
        event_id, vac_idx, dir_idx = self._decode_event(action)
        current_base_rates = self._get_cached_base_rates_np(force=False)
        current_event_features = self.get_event_features(force=False, copy=False)
        rates = current_base_rates.astype(np.float64, copy=False)
        total_rate = float(np.sum(rates))
        selected_rate = float(rates[vac_idx, dir_idx]) if total_rate > 0.0 else 0.0
        legal_mask = current_event_features[..., 0] > 0.5
        best_available_energy_gain = float(np.max(current_event_features[..., 4][legal_mask])) if np.any(legal_mask) else 0.0
        selected_energy_gain = 0.0
        if 0 <= vac_idx < current_event_features.shape[0] and 0 <= dir_idx < current_event_features.shape[1]:
            if bool(current_event_features[vac_idx, dir_idx, 0] > 0.5):
                selected_energy_gain = float(current_event_features[vac_idx, dir_idx, 4])
        missed_energy_gain = max(best_available_energy_gain - selected_energy_gain, 0.0)
        legal_event_count = int(np.sum(legal_mask))

        energy_before = float(self.energy_last)

        self.action_history_array[:, :-1] = self.action_history_array[:, 1:]
        self.action_history_array[:, -1] = -1
        if 0 <= vac_idx < self.num_agents:
            self.action_history_array[vac_idx, -1] = dir_idx if selected_rate > 0.0 else -1

        if total_rate > 0.0 and selected_rate > 0.0:
            event_info = self._apply_single_event(vac_idx, dir_idx)
            delta_t = 0.0  # Policy event: physical clock is unassigned.
            self.time += delta_t
        else:
            event_info = {
                "selected_vacancy": vac_idx,
                "selected_direction": dir_idx,
                "old_pos": tuple(map(int, self.vacancy_pos[vac_idx])),
                "new_pos": tuple(map(int, self.vacancy_pos[vac_idx])),
                "moving_type": VACANCY,
                "legal_event": False,
                "delta_e_exact": 0.0,
                "local_energy_change": 0.0,
            }
            delta_t = 0.0

        if event_info["legal_event"]:
            delta_e = float(event_info["delta_e_exact"])
            energy_after = energy_before + delta_e
        else:
            energy_after = energy_before

        self._env_step_counter += 1
        if self.energy_check_interval > 0 and (self._env_step_counter % self.energy_check_interval == 0):
            full_energy = float(self.calculate_system_energy())
            if abs(full_energy - energy_after) > self.energy_check_tol:
                energy_after = full_energy

        global_energy_change = energy_before - energy_after
        reward_scale = float(getattr(self.args, "reward_scale", 1.0))
        reward = reward_scale * global_energy_change

        reward_components = {
            "energy_reward": float(reward),
            "step_penalty": float(getattr(self.args, "step_penalty", 0.0)),
            "move_bonus": 0.0,
            "illegal_penalty": 0.0,
            "new_best_bonus": 0.0,
            "best_event_gap_penalty": 0.0,
        }

        reward += reward_components["step_penalty"]
        if event_info["legal_event"]:
            if int(event_info["moving_type"]) in (AL, CR):
                reward_components["move_bonus"] = float(
                    getattr(self.args, "solute_move_bonus", getattr(self.args, "cu_move_bonus", 0.0))
                )
            elif int(event_info["moving_type"]) == NI:
                reward_components["move_bonus"] = float(
                    getattr(self.args, "matrix_move_penalty", getattr(self.args, "fe_move_penalty", 0.0))
                )
            reward += reward_components["move_bonus"]

            best_improvement = 0.0 if self.energy_best is None else max(float(self.energy_best) - float(energy_after), 0.0)
            if best_improvement > 0.0:
                reward_components["new_best_bonus"] = float(getattr(self.args, "new_best_energy_bonus", 0.0)) * best_improvement
                reward += reward_components["new_best_bonus"]
                self.energy_best = float(energy_after)
        else:
            reward_components["illegal_penalty"] = float(getattr(self.args, "illegal_action_penalty", 0.0))
            reward += reward_components["illegal_penalty"]

        missed_gain_coef = float(getattr(self.args, "best_event_gap_penalty_coef", 0.0))
        if missed_gain_coef > 0.0 and missed_energy_gain > 0.0:
            reward_components["best_event_gap_penalty"] = -missed_gain_coef * missed_energy_gain
            reward += reward_components["best_event_gap_penalty"]

        self.energy_last = energy_after
        self.energy_history.append(energy_after)
        self.time_history.append(self.time)

        done = bool(total_rate <= 0.0)
        self._refresh_after_event(event_info)

        obs = self.get_combined_obs().astype(np.float32)
        share_stats = self.get_system_stats(force=done).astype(np.float32)
        full_obs = self._build_full_obs(obs, share_stats)
        positions = np.asarray(self.vacancy_pos, dtype=np.int32)

        next_base_rates = self._get_cached_base_rates_np(force=False)
        next_event_features = self.get_event_features(force=False, copy=False)
        next_sparse_defect_tokens, next_sparse_defect_masks = self.get_sparse_defect_context(force=False, copy=False)

        rewards = np.array([[reward]], dtype=np.float32)
        dones = np.array([done], dtype=np.bool_)
        infos = {
            "event_id": int(event_id),
            "selected_vacancy": int(event_info["selected_vacancy"]),
            "selected_direction": int(event_info["selected_direction"]),
            "selected_rate": float(selected_rate),
            "total_rate": float(total_rate),
            "legal_event": bool(event_info["legal_event"]),
            "moving_type": int(event_info["moving_type"]),
            "old_pos": event_info["old_pos"],
            "new_pos": event_info["new_pos"],
            "delta_e_exact": float(event_info["delta_e_exact"]),
            "local_energy_change": float(event_info["local_energy_change"]),
            "global_energy_change": float(global_energy_change),
            "total_energy": float(energy_after),
            "total_system_energy": float(energy_after),
            "kmc_time": float(self.time),
            "kmc_delta_t": float(delta_t),
            "clock_owner": "unassigned",
            "physical_time_validated": False,
            "next_base_rates": next_base_rates,
            "next_event_features": next_event_features,
            "next_sparse_defect_tokens": next_sparse_defect_tokens,
            "next_sparse_defect_masks": next_sparse_defect_masks,
            "legal_event_count": legal_event_count,
            "selected_energy_gain": float(selected_energy_gain),
            "best_available_energy_gain": float(best_available_energy_gain),
            "missed_energy_gain": float(missed_energy_gain),
            "reward_components": reward_components,
        }
        return obs, full_obs, positions, rewards, dones, infos

    def get_states(self):
        device = torch.device(getattr(self.args, 'device', 'cpu'))
        local_env = torch.tensor(self.static_vancancy_local_atoms(), dtype=torch.int64, device=device)
        return local_env

    def get_base_rates(self):
        device = torch.device(getattr(self.args, 'device', 'cpu'))
        base_rates = torch.tensor(self.get_base_rates_np(force=False, copy=False), dtype=torch.float32, device=device)
        return base_rates

    def get_event_features_tensor(self):
        device = torch.device(getattr(self.args, 'device', 'cpu'))
        return torch.tensor(self.get_event_features(force=False, copy=False), dtype=torch.float32, device=device)
