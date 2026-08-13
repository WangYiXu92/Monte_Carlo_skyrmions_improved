"""MX2 单层（1T-CrX2 型）磁滞回线演示。

MX2 单层中磁性 M 原子形成三角子格。这里用"磁性子格有效模型"：
单 basis 三角格 + 最近邻交换 + 界面 DMI + 易轴各向异性。
三个例子的分工：root = skyrmion 退火，CrI3_test_curie = 居里温度，
本目录 = 磁滞回线。
"""
import numpy as np

from mc import Lattice, Hamiltonian, MonteCarlo2D

# =====================================================================
# 0. 选择功能
# =====================================================================
SIMULATION_MODE = "hysteresis"
SEED = 20260813          # 随机种子（None = 不固定）

# =====================================================================
# 1. 晶格参数：三角格（M 子格）
# =====================================================================
a = 1.0
a_vecs = [[a, 0.0], [0.5 * a, np.sqrt(3.0) / 2.0 * a]]
basis = [[0.0, 0.0]]
Nx, Ny = 60, 60

# =====================================================================
# 2. 哈密顿量参数（能量单位均为 meV）
# =====================================================================
A_ani = -0.05
J_ex = -0.5
D = 0.15


def build_simulator():
    """建立晶格、哈密顿量和随机初态。"""
    lattice = Lattice(a_vecs, basis, Nx, Ny)
    ham = Hamiltonian(Nb=len(basis), A_ani=A_ani)

    # 三角格 3 条最近邻 bond；矩阵反对称部分 = 界面型 DMI（d = D·ẑ×r̂）。
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
# 3. 磁滞回线参数
#     B_list 的顺序就是扫描路径；必须包含正扫和反扫才会形成回线。
#     交换能标度 ~6×|J|/2 = 1.5 meV；μ_s·15 T ≈ 1.74 meV > 1.5
#     → 场能足以翻转磁序，回线有真实翻转（原版 J=-8 时场永远翻不动）。
# =====================================================================
HYSTERESIS_PARAMS = {
    "B_list": np.concatenate((
        np.linspace(15.0, -15.0, 61),
        np.linspace(-15.0, 15.0, 61)[1:],
    )),                         # Tesla
    "T": 5.0,                   # K
    "equip_steps": 1500,        # 每个场点先平衡的 sweep 数
    "calc_steps": 1500,         # 每个场点的记录次数
    "sample_interval": 1,
    "output_file": "hysteresis_loop.txt",
}

if __name__ == "__main__":
    mc = build_simulator()
    if SIMULATION_MODE == "hysteresis":
        mc.run_hysteresis_loop(**HYSTERESIS_PARAMS)
    else:
        raise ValueError("本例子只演示 hysteresis 模式")
