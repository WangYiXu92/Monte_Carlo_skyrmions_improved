"""Skyrmion 分析 demo：定位 → 统计 → 热稳定性曲线。

用法：python skyrmion_analysis_demo.py  → 输出 skyrmion_stability.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc import Lattice, Hamiltonian, MonteCarlo2D

# 三角格，J=-40 meV, D=4.0（D/J=0.1，skyrmion 相稳定窗口）, A=-0.04 易轴, B=1T
a = 1.0
a_vecs = [[a, 0.0], [0.5*a, np.sqrt(3.0)/2.0*a]]
lat = Lattice(a_vecs, [[0, 0]], 42, 42)
ham = Hamiltonian(Nb=1, A_ani=-0.04)
J_ex, D = -40.0, 4.0
ham.add_bond(0, 0, [1, 0],   [[J_ex,0,-D],[0,J_ex,0],[D,0,J_ex]])
ham.add_bond(0, 0, [0, 1],   [[J_ex,0,-0.5*D],[0,J_ex,-np.sqrt(3)/2*D],[0.5*D,np.sqrt(3)/2*D,J_ex]])
ham.add_bond(0, 0, [-1, 1],  [[J_ex,0,0.5*D],[0,J_ex,-np.sqrt(3)/2*D],[-0.5*D,np.sqrt(3)/2*D,J_ex]])
mc = MonteCarlo2D(lat, ham, seed=42)

def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def sk_unit(x, y, x0, y0, lam=3.5):
    dx, dy = x - x0, y - y0
    r2 = dx*dx + dy*dy
    return norm(np.array([2*lam*dy/(r2+lam*lam), -2*lam*dx/(r2+lam*lam),
                          (r2-lam*lam)/(r2+lam*lam)]))

# 构造 14×14 周期 skyrmion 晶格（9 个）→ 极低温弛豫
for x in range(42):
    for y in range(42):
        x0 = 14.0*(x // 14) + 7.0
        y0 = 14.0*(y // 14) + 7.0
        mc.spins[x, y, 0] = sk_unit(x, y, x0, y0)
mc.ham.B_field_meV = np.array([0.0, 0.0, 1.0]) * 0.1157735
for _ in range(200):
    mc.mc_step(0.01)

st = mc.skyrmion_statistics()
print(f"弛豫后: {st['n_skyrmions']} 个 skyrmion, 半径 {st['radius_mean']:.2f}±{st['radius_std']:.2f}, "
      f"晶格常数 {st['lattice_constant']:.2f}, Q = {st['total_Q']:.2f}")

# 升温稳定性扫描
T_list = [3, 6, 9, 12, 15, 18, 22, 26, 30, 36, 44, 55]
Ts, Ns, Ns_std = mc.skyrmion_stability(T_list, equip_steps=250, calc_steps=30,
                                        output_file="skyrmion_stability.txt")

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(Ts, Ns, yerr=Ns_std, fmt="o-", ms=6, lw=1.5, capsize=3)
ax.set_xlabel("T (K)")
ax.set_ylabel(r"$N_{\rm skyrmion}$")
ax.set_title("Skyrmion lattice thermal stability (J=-40, D=4, B=1T)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("skyrmion_stability.png", dpi=200)
print("已保存 skyrmion_stability.png")
