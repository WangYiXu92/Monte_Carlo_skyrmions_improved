#!/usr/bin/env python3
"""CrI3 honeycomb 等温磁化 M(B) + 热磁 M(T) 曲线 demo。

协议：B 外循环（0→10T），T 内循环降序冷却——零场冷却后升场（FC 式）。
出图：左 M vs B（等温磁化，每 T 一条），右 M vs T（热磁曲线，每 B 一条）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mc import Lattice, Hamiltonian, MonteCarlo2D

# ---- CrI3 honeycomb：J = -6 meV（最近邻，3 方向），Tc ≈ 37 K ----
lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 24, 24)
ham = Hamiltonian(Nb=2, A_ani=-0.5)
for off in ([0, 0], [-1, 0], [0, -1]):
    ham.add_bond(0, 1, off, np.diag([-6.0] * 3))

T_list = [5.0, 10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 70.0]
B_list = np.arange(0.0, 10.5, 1.0)          # 0..10 T

mc = MonteCarlo2D(lat, ham, seed=20260814)
T_list, B_list, M = mc.run_magnetization_curves(
    T_list, B_list, equip_steps=8000, calc_steps=4000,
    sample_interval=4, output_csv="MCE_demo/magnetization_curves.txt")

# ---- 图：左 M(B) 等温、右 M(T) 热磁 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cmap = plt.cm.viridis(np.linspace(0, 1, len(T_list)))
for i, T in enumerate(T_list):
    axes[0].plot(B_list, M[i], "o-", ms=4, lw=1.5, color=cmap[i], label=f"{T:.0f} K")
axes[0].set_xlabel("B (T)"); axes[0].set_ylabel(r"$M$ (per spin $S_z$)")
axes[0].set_title("Isothermal magnetization M(B)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
cmap2 = plt.cm.plasma(np.linspace(0, 1, len(B_list)))
for j, B in enumerate(B_list):
    axes[1].plot(T_list, M[:, j], "o-", ms=4, lw=1.5, color=cmap2[j], label=f"{B:.0f} T")
axes[1].set_xlabel("T (K)"); axes[1].set_ylabel(r"$M$ (per spin $S_z$)")
axes[1].set_title("Thermomagnetic curves M(T)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("MCE_demo/magnetization_curves.png", dpi=150)
print("已保存 MCE_demo/magnetization_curves.png")
