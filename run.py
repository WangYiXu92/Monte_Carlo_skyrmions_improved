"""二维自旋 Monte Carlo 的唯一运行入口。

只需修改 ``SIMULATION_MODE`` 和相应参数，然后运行：
    python run.py

可选模式：
    "skyrmion"   - 固定磁场下退火，输出 skyrmion_spins.txt
    "curie"      - 固定磁场下温度扫描，输出 curie_results.txt
    "hysteresis" - 沿给定磁场路径扫描，输出 hysteresis_loop.txt
"""

import numpy as np

from mc import Lattice, Hamiltonian, MonteCarlo2D


# =====================================================================
# 0. 选择要运行的功能："skyrmion"、"curie" 或 "hysteresis"
# =====================================================================
SIMULATION_MODE = "skyrmion"
SEED = 20260813          # 随机种子（None = 不固定，结果不可复现）


# =====================================================================
# 1. 晶格参数：二维六角 Bravais 晶格
#    a1 = (a, 0), a2 = (a/2, sqrt(3)*a/2)
#    basis 坐标为分数坐标 (f1, f2)，实际位置为
#    r = (n1+f1)*a1 + (n2+f2)*a2。
# =====================================================================
a = 1.0
a_vecs = [[a, 0.0], [0.5 * a, np.sqrt(3.0) / 2.0 * a]]
basis = [[0.0, 0.0]]
Nx, Ny = 90, 90


# =====================================================================
# 2. 哈密顿量参数（能量单位均为 meV）
#    本例为最近邻铁磁交换 + 界面型 DMI + 易轴各向异性。
# =====================================================================
A_ani = -0.04
J_ex = -10.0
D = 1.45

def build_simulator():
    """建立晶格、哈密顿量和随机初态。"""
    lattice = Lattice(a_vecs, basis, Nx, Ny)
    ham = Hamiltonian(Nb=len(basis), A_ani=A_ani)

    # 每条物理 bond 只输入一次；add_bond 会自动补上反向 bond。
    # 矩阵的对称部分为交换作用，反对称部分为 DMI。
    J_mat_10 = np.array([
        [J_ex, 0.0, -D],
        [0.0, J_ex, 0.0],
        [D, 0.0, J_ex],
    ])
    ham.add_bond(0, 0, [1, 0], J_mat_10)

    J_mat_01 = np.array([
        [J_ex, 0.0, -0.5 * D],
        [0.0, J_ex, -np.sqrt(3.0) / 2.0 * D],
        [0.5 * D, np.sqrt(3.0) / 2.0 * D, J_ex],
    ])
    ham.add_bond(0, 0, [0, 1], J_mat_01)

    J_mat_m11 = np.array([
        [J_ex, 0.0, 0.5 * D],
        [0.0, J_ex, -np.sqrt(3.0) / 2.0 * D],
        [-0.5 * D, np.sqrt(3.0) / 2.0 * D, J_ex],
    ])
    ham.add_bond(0, 0, [-1, 1], J_mat_m11)

    return MonteCarlo2D(lattice, ham, seed=SEED)


# =====================================================================
# 3A. Skyrmion 退火参数
# =====================================================================
SKYRMION_PARAMS = {
    "T_init": 300.0,          # K
    "T_final": 0.05,          # K
    "steps_per_T": 2000,      # 每个退火温度的 sweep 数
    "B_field": [0.0, 0.0, 1.0],  # Tesla，通常可尝试非零 z 场
}


# =====================================================================
# 3B. 温度扫描 / 居里温度参数
#     B_field 可设为非零，以研究指定磁场下的磁性变化。
# =====================================================================
CURIE_PARAMS = {
    "T_list": np.linspace(1.0, 80.0, 30),  # K
    "equip_steps": 3000,       # 每个温度下先平衡的 sweep 数
    "calc_steps": 4000,        # 每个温度下的记录次数
    "sample_interval": 1,      # 相邻记录之间的 sweep 数
    "B_field": [0.0, 0.0, 0.0],  # Tesla；例如 [0, 0, 1.0]
    "output_file": "curie_results.txt",
}


# =====================================================================
# 3C. 磁滞回线参数
#     B_list 的顺序就是扫描路径；必须包含正扫和反扫才会形成回线。
# =====================================================================
HYSTERESIS_PARAMS = {
    "B_list": np.concatenate((
        np.linspace(10.0, -10.0, 41),
        np.linspace(-10.0, 10.0, 41)[1:],
    )),                         # Tesla
    "T": 1.0,                  # K
    "equip_steps": 1000,       # 每个场点先平衡的 sweep 数
    "calc_steps": 1000,        # 每个场点的记录次数
    "sample_interval": 1,
    "output_file": "hysteresis_loop.txt",
}


if __name__ == "__main__":
    mc = build_simulator()

    if SIMULATION_MODE == "skyrmion":
        mc.run_skyrmion_annealing(**SKYRMION_PARAMS)
    elif SIMULATION_MODE == "curie":
        mc.run_curie_temperature(**CURIE_PARAMS)
    elif SIMULATION_MODE == "hysteresis":
        mc.run_hysteresis_loop(**HYSTERESIS_PARAMS)
    else:
        raise ValueError(
            "SIMULATION_MODE 必须是 'skyrmion'、'curie' 或 'hysteresis'"
        )
