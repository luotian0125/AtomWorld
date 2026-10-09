from typing import Tuple
import numpy as np

class Lattice:
    def __init__(self, args):
        self.size = args.lattice_size
        self.Cu_nums = args.lattice_cu_nums
        self.V_nums = args.lattice_v_nums
        self.NN1 = np.array([
            [1, 1, 1],  [-1, -1, -1],  
            [1, 1, -1], [-1, -1, 1],   
            [1, -1, 1], [-1, 1, -1],   
            [-1, 1, 1], [1, -1, -1]    
        ], dtype=int)
        self.NN2 = np.array([
            [2, 0, 0], [-2, 0, 0], 
            [0, 2, 0], [0, -2, 0],
            [0, 0, 2], [0, 0, -2]
        ], dtype=int)

        # 【新增】3NN (12个): <2, 2, 0> 族 -> 距离 sqrt(8)
        self.NN3 = np.array([
            [2, 2, 0], [2, -2, 0], [-2, 2, 0], [-2, -2, 0],
            [2, 0, 2], [2, 0, -2], [-2, 0, 2], [-2, 0, -2],
            [0, 2, 2], [0, 2, -2], [0, -2, 2], [0, -2, -2]
        ], dtype=int)

        # 【新增】4NN (24个): <3, 1, 1> 族 -> 距离 sqrt(11)
        # 这一步稍微繁琐，需要生成所有排列组合
        nn4_list = []
        for x in [3, -3]:
            for y in [1, -1]:
                for z in [1, -1]:
                    nn4_list.append([x, y, z]) # (3, 1, 1) permutations
                    nn4_list.append([y, x, z]) # (1, 3, 1) permutations
                    nn4_list.append([y, z, x]) # (1, 1, 3) permutations
        self.NN4 = np.array(sorted(list(set(map(tuple, nn4_list))),
                    key=lambda x: (x[0], x[1], x[2])), dtype=int)
        
        self.coords = self.generate_lattice_coordinates()
        self.lattice = None
        self.vacancy_pos = None
        self.cu_pos = None
        self.cu_pos_index = None
        self.init_lattice()

    def generate_lattice_coordinates(self):
        nx, ny, nz = self.size
        corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
        body = corner + 1
        return np.vstack((corner, body))

    def _build_cu_pos_index(self):
            self.cu_pos_index = {tuple(pos): idx for idx, pos in enumerate(self.cu_pos)} 

    def update_cu_pos(self):
        nx, ny, nz = self.size
        corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
        body = corner + 1
        all_coords = np.vstack((corner, body))
        atom_types = self.lattice[tuple(all_coords.T)]
        self.cu_pos = all_coords[atom_types == 1].tolist()



    def init_lattice(self):
        Cu_nums = self.Cu_nums
        V_nums = self.V_nums
        nx, ny, nz = self.size
        lattice_shape = (nx * 2, ny * 2, nz * 2)
        total_sites = nx * ny * nz * 2
        assert Cu_nums + V_nums <= total_sites, "原子数超出晶格容量"
        atom_list = np.array(
            [1] * Cu_nums + [2] * V_nums + [0] * (total_sites - Cu_nums - V_nums),
            dtype=np.int32
        )
        np.random.shuffle(atom_list)

        corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
        body = corner + 1
        all_coords = np.vstack((corner, body))

        lattice = np.full(lattice_shape, -1, dtype=np.int32)
        lattice[tuple(all_coords.T)] = atom_list

        vacancy_pos = all_coords[atom_list == 2]
        cu_pos = all_coords[atom_list == 1]

        self.lattice = lattice
        self.vacancy_pos = vacancy_pos
        self.cu_pos = cu_pos
        self._build_cu_pos_index()
            

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    class Args:
        lattice_size = (4, 4, 4)
        lattice_cu_nums = 20
        lattice_v_nums = 10

    args = Args()
    lattice_obj = Lattice(args)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    lattice = lattice_obj.lattice

    coords = lattice_obj.coords
    atom_types = lattice[tuple(coords.T)]

    coords = np.array(coords)
    Fe_coords = coords[atom_types == 0]
    Cu_coords = coords[atom_types == 1]
    Vac_coords = coords[atom_types == 2]

    ax.scatter(Fe_coords[:, 0], Fe_coords[:, 1], Fe_coords[:, 2], c='gray', label='Fe', s=10)
    ax.scatter(Cu_coords[:, 0], Cu_coords[:, 1], Cu_coords[:, 2], c='orange', label='Cu', s=30)
    ax.scatter(Vac_coords[:, 0], Vac_coords[:, 1], Vac_coords[:, 2], c='red', label='Vacancy', s=50, marker='x')

    ax.set_title("Lattice Structure Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()
    # plt.savefig("./plots/lattice_structure.png")
