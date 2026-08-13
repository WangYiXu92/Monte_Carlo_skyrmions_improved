"""CrI3 单层居里温度扫描（honeycomb 双子格，修正版）。

真实 CrI3 的 Cr 子格是 honeycomb（双子格、每自旋 3 个最近邻），
不是三角格。本例子使用标准 honeycomb 表示：
    basis = [[1/3, 1/3], [2/3, 2/3]]
    3 条 A→B 最近邻 bond：(0,0), (-1,0), (0,-1)
参数取文章原值 J = -6 meV（FM）, A = -0.5 meV（易 z 轴）, D = 0。
注意：经典 Heisenberg 2D 无各向异性时没有有限温长程序，
Tc 由单离子各向异性 A 撑起——A 越大 Tc 越高。
"""
import numpy as np

from mc import Lattice, Hamiltonian, MonteCarlo2D

# =====================================================================
# 0. 选择功能
# =====================================================================
SIMULATION_MODE = "curie"
SEED = 20260813          # 随机种子（None = 不固定）

# =====================================================================
# 1. 晶格参数：honeycomb（六角 Bravais + 双子格）
# =====================================================================
a = 1.0
a_vecs = [[a, 0.0], [0.5 * a, np.sqrt(3.0) / 2.0 * a]]
basis = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
Nx, Ny = 60, 60

# =====================================================================
# 2. 哈密顿量参数（能量单位均为 meV）
# =====================================================================
A_ani = -0.5
J_ex = -6.0
D = 0.0


def build_simulator():
    """建立晶格、哈密顿量和随机初态。"""
    lattice = Lattice(a_vecs, basis, Nx, Ny)
    ham = Hamiltonian(Nb=len(basis), A_ani=A_ani)

    # 每条物理 bond 只输入一次；add_bond 会自动补上反向 bond（J^T）。
    # honeycomb 的 3 条最近邻 bond 从子格 0 指向子格 1。
    for off in ([0, 0], [-1, 0], [0, -1]):
        J_mat = np.array([
            [J_ex, 0.0, 0.0],
            [0.0, J_ex, 0.0],
            [0.0, 0.0, J_ex],
        ])
        ham.add_bond(0, 1, off, J_mat)

    return MonteCarlo2D(lattice, ham, seed=SEED)


# =====================================================================
# 3. 温度扫描 / 居里温度参数
# =====================================================================
CURIE_PARAMS = {
    "T_list": np.linspace(1.0, 100.0, 40),  # K
    "equip_steps": 2000,       # 每个温度下先平衡的 sweep 数
    "calc_steps": 3000,        # 每个温度下的记录次数
    "sample_interval": 2,      # 相邻记录之间的 sweep 数（降自相关）
    "B_field": [0.0, 0.0, 0.0],
    "output_file": "curie_results.txt",
}

if __name__ == "__main__":
    mc = build_simulator()
    if SIMULATION_MODE == "curie":
        mc.run_curie_temperature(**CURIE_PARAMS)
    else:
        raise ValueError("本例子只演示 curie 模式")
