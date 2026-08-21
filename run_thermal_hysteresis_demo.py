#!/usr/bin/env python3
"""Thermal hysteresis demo: T-loop at fixed field inside the spin-flop bistable window.

Physics: easy-axis AFM (J=+1, A=-1.0) at B=20.5 T — just below B_sf≈21.2 T
(spin-flip barrier 2|A|=2 meV ≫ k_B·T here).
Two metastable states (Neel along z vs xy-flop) are separated by a barrier:
- heating from low T (Neel) : barrier crossed at T_up
- cooling from high T (flop): barrier crossed at T_down
If T_up != T_down → thermal hysteresis in M_z(T).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mc import Lattice, Hamiltonian, MonteCarlo2D, MU_S_MEV_PER_T

lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 24, 24)
ham = Hamiltonian(Nb=2, A_ani=-1.0)
for off in ([0, 0], [-1, 0], [0, -1]):
    ham.add_bond(0, 1, off, np.diag([1.0] * 3))
mc = MonteCarlo2D(lat, ham, seed=20260814)

B_FIX = 20.5           # T — just below B_sf≈21.2T (A=-1: barrier 2|A|=2 meV ≫ k_B T)
T_LO, T_HI, dT = 2.0, 40.0, 2.0
EQUIP, CALC = 8000, 4000

# Start from a Neel state: equilibrate at B=4 T (deep in Neel) then quench to B_FIX
mc.ham.B_field_meV = np.array([0.0, 0.0, 4.0], dtype=np.float64) * MU_S_MEV_PER_T
for _ in range(20000):
    mc.mc_step(2.0)
mc.ham.B_field_meV = np.array([0.0, 0.0, B_FIX], dtype=np.float64) * MU_S_MEV_PER_T
for _ in range(8000):
    mc.mc_step(2.0)
print(f"起始态 (B={B_FIX} T, T=2K): M_z = {mc.get_magnetization()[1][2]:.3f}")

def scan(Ts, label):
    out = []
    for T in Ts:
        for _ in range(EQUIP):
            mc.mc_step(T)
        m_acc = 0.0
        for _ in range(CALC):
            for _ in range(4):
                mc.mc_step(T)
            m_acc += mc.get_magnetization()[1][2]
        out.append([T, m_acc / CALC])
        print(f"  [{label}] T={T:5.1f} K  M_z={out[-1][1]:.3f}")
    return np.array(out)

print("--- heating 2→40 K ---")
up = scan(np.arange(T_LO, T_HI + 1e-9, dT), "up")
print("--- cooling 40→2 K ---")
dn = scan(np.arange(T_HI, T_LO - 1e-9, -dT), "dn")

np.savetxt("MCE_demo/thermal_hyst_up.txt", up, header="T_K Mz")
np.savetxt("MCE_demo/thermal_hyst_down.txt", dn, header="T_K Mz")

# 热滞判据：两路径 M_z 差最大的 T 区间宽度（|M_up(T)−M_dn(T)|>0.2 的 T 数）
# ⚠️ np.interp 要求 x 升序——dn 是降序，必须先反转
dn_s = dn[::-1]
B_common = np.intersect1d(up[:, 0], dn_s[:, 0])
dM = [abs(np.interp(t, up[:, 0], up[:, 1]) - np.interp(t, dn_s[:, 0], dn_s[:, 1])) for t in B_common]
hyst_T = [t for t, dm in zip(B_common, dM) if dm > 0.2]
if len(hyst_T) > 1:
    print(f"\n热滞区间（|ΔM_z|>0.2）: {hyst_T[0]:.0f} ~ {hyst_T[-1]:.0f} K, 宽度 = {hyst_T[-1]-hyst_T[0]:.1f} K")
elif len(hyst_T) == 1:
    print(f"\n热滞区间: 单点 {hyst_T[0]:.0f} K（非热滞，转变 T 相同）")
else:
    print("\n无热滞（两路径 |ΔM_z| ≤ 0.2）——势垒对称时 spin-flop 的 T 扫描不显示热滞")

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(up[:, 0], up[:, 1], "o-", ms=4, lw=1.5, color="tab:red", label=f"heating (T: 2→{T_HI:.0f} K)")
ax.plot(dn[:, 0], dn[:, 1], "s-", ms=4, lw=1.5, color="tab:blue", label=f"cooling (T: {T_HI:.0f}→2 K)")
ax.set_xlabel("T (K)"); ax.set_ylabel(r"$M_z$ (per spin)")
ax.set_title(f"Thermal hysteresis at B={B_FIX} T (AFM spin-flop, J=+1, A=$-1.0$)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("MCE_demo/thermal_hysteresis.png", dpi=150)
print("已保存 MCE_demo/thermal_hysteresis.png")
