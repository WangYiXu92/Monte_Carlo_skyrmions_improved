#!/usr/bin/env python3
"""AFM spin-flop demo：一阶场致跃迁验证（固定晶格 ≠ 只能算二阶）。

物理：易轴(z)反铁磁（honeycomb, J=+1.0 meV AFM, A_ani=-0.25 meV 易轴）加垂直场 B_z。
- B < B_sf：Neél 序沿 z，M_z ≈ 0
- B ≈ B_sf ≈ (1/μ_s)·√(2|A|·z·J) ≈ 10.6 T：自旋翻转到 xy 面（spin-flop），M_z 跳变——一阶
- 回扫时在 B_sf 处回滞（不可逆跃迁的直接证据）
验证：跳变前后采样 ⟨S_x⟩/⟨S_y⟩/⟨S_z⟩——flop 态应面内（S_xy 大、S_z 小）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mc import Lattice, Hamiltonian, MonteCarlo2D

# ---- AFM honeycomb：J=+1.0（AFM 约定 J>0），A=-0.25（易轴 z）----
lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 24, 24)
ham = Hamiltonian(Nb=2, A_ani=-0.25)
for off in ([0, 0], [-1, 0], [0, -1]):
    ham.add_bond(0, 1, off, np.diag([1.0] * 3))

T = 2.0
B_up = np.arange(0.0, 20.5, 0.5)          # 上扫 0→20 T
B_down = np.arange(19.5, -0.25, -0.5)     # 下扫 20→0 T

mc = MonteCarlo2D(lat, ham, seed=20260814)
res_up = mc.run_hysteresis_loop(B_up, T, equip_steps=6000, calc_steps=3000,
                                sample_interval=4, output_file="MCE_demo/spinflop_up.txt")
res_dn = mc.run_hysteresis_loop(B_down, T, equip_steps=6000, calc_steps=3000,
                                sample_interval=4, output_file="MCE_demo/spinflop_down.txt")

# ---- spin-flop 判定：跳变前后自旋方向 ----
print("\n--- 自旋方向采样（Neél vs flop）---")
for Bs in (4.0, 9.0, 11.0, 13.0, 18.0):
    mc.ham.B_field_meV = np.array([0.0, 0.0, Bs], dtype=np.float64) * 0.1157676
    for _ in range(4000):
        mc.mc_step(T)
    acc = np.zeros(3)
    for _ in range(2000):
        for _ in range(4):
            mc.mc_step(T)
        acc += mc.get_magnetization()[1]
    S_avg = acc / 2000
    print(f"  B={Bs:5.1f} T: <S_x>={S_avg[0]:+.3f} <S_y>={S_avg[1]:+.3f} <S_z>={S_avg[2]:+.3f}  "
          f"面内占比={np.hypot(S_avg[0],S_avg[1]):.3f}")
mc.ham.B_field_meV = np.zeros(3)

# ---- 图：上扫/下扫 M_z(B)，标注 spin-flop 回滞 ----
dU = np.loadtxt("MCE_demo/spinflop_up.txt", skiprows=1)
dD = np.loadtxt("MCE_demo/spinflop_down.txt", skiprows=1)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(dU[:, 0], dU[:, 1], "o-", ms=4, lw=1.5, color="tab:blue", label="up sweep (0→20 T)")
ax.plot(dD[:, 0], dD[:, 1], "s-", ms=4, lw=1.5, color="tab:red", label="down sweep (20→0 T)")
# 回滞宽度（找最大 |M_up−M_dn| 同 B 差）
B_common = np.intersect1d(dU[:, 0], dD[:, 0])
hyst = max(abs(np.interp(b, dU[:, 0], dU[:, 1]) - np.interp(b, dD[:, 0], dD[:, 1])) for b in B_common)
print(f"\n最大回滞宽度 ΔM = {hyst:.3f}（>0.1 = 一阶跃迁特征）")
ax.set_xlabel("B (T)"); ax.set_ylabel(r"$M_z$ (per spin)")
ax.set_title(f"AFM spin-flop, T={T} K (J=+1 meV, A=$-0.25$ meV easy-axis)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("MCE_demo/spinflop.png", dpi=150)
print("已保存 MCE_demo/spinflop.png")
