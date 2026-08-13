"""CrI3 单层居里温度扫描（honeycomb 双子格，修正版）。

真实 CrI3 的 Cr 子格是 honeycomb（双子格、每自旋 3 个最近邻），
不是三角格。本例子使用标准 honeycomb 表示：
    basis = [[1/3, 1/3], [2/3, 2/3]]
    3 条 A→B 最近邻 bond：(0,0), (-1,0), (0,-1)
参数取文章原值 J = -6 meV（FM）, A = -0.5 meV（易 z 轴）, D = 0。

扫描方向：从高温向低温（cooling）。从随机初态直接加热时，
低 T 平衡极慢（接受率低），曲线会被"未平衡起点"污染（原版
heating 扫描 M(1K)≈0.2 而非饱和值）。降温扫描每个 T 从上一
温度的末态出发，低 T 时系统已有序，曲线干净。

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
Nx, Ny = 30, 30   # 30×30=1800 spins：平衡充分且速度快（60×60 的畴壁生长需要 ~10× 更多 sweeps）

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
#     cooling：从 100 K 向 1 K 扫描（低 T 平衡更充分）
# =====================================================================
CURIE_PARAMS = {
    "T_list": np.linspace(100.0, 1.0, 40),  # K（降温方向）
    "equip_steps": 5000,       # 每个温度下先平衡的 sweep 数（60×60 需 ~10× 更多）
    "calc_steps": 5000,        # 每个温度下的记录次数
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
