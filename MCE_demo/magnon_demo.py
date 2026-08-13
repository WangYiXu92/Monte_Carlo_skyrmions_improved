"""磁振子色散 demo：honeycomb CrI3 型（FM, J=-6, A=-0.5, S=1.5）。

用法：python magnon_demo.py  → 输出 magnon_band.png（Γ-K-M-Γ 路径）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc import Lattice, Hamiltonian, MonteCarlo2D

a = 1.0
a_vecs = [[a, 0.0], [0.5 * a, np.sqrt(3.0) / 2.0 * a]]
basis = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

lat = Lattice(a_vecs, basis, 8, 8)
ham = Hamiltonian(Nb=2, A_ani=-0.5)
for off in ([0, 0], [-1, 0], [0, -1]):
    ham.add_bond(0, 1, off, np.diag([-6.0] * 3))
mc = MonteCarlo2D(lat, ham, seed=1)

# Γ-K-M-Γ 路径（分数坐标，每段 40 点）
seg = [(0, 0), (1 / 3, 1 / 3), (0.5, 0), (0, 0)]
nseg = 40
kpath, xpos, labels = [], [0.0], ["Γ"]
for i in range(len(seg) - 1):
    k1, k2 = seg[i], seg[i + 1]
    for j in range(nseg):
        t = j / nseg
        kpath.append((k1[0] + t * (k2[0] - k1[0]), k1[1] + t * (k2[1] - k1[1])))
    # 段长（k 空间距离，用倒格矢）
    dk = np.array(k2) - np.array(k1)
    xpos.append(xpos[-1] + np.hypot(dk[0], dk[1]))
    labels.append(["K", "M", "Γ"][i])
xpos = np.array(xpos)

_, w = mc.magnon_spectrum(kpath, S=1.5)

# 等距 x 轴（段内线性，段界标注）
xs = np.linspace(0, len(kpath) - 1, len(kpath))
fig, ax = plt.subplots(figsize=(6, 4))
for b in range(w.shape[1]):
    ax.plot(xs, w[:, b], "-", lw=1.5, label=f"band {b+1}")
ax.set_xticks(xs[[0, nseg - 1, 2 * nseg - 1, 3 * nseg - 1]])
ax.set_xticklabels(labels)
ax.set_ylabel(r"$\omega$ (meV)")
ax.set_title("Magnon dispersion: honeycomb FM J=-6 meV, A=-0.5 meV, S=1.5")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("magnon_band.png", dpi=200)
print("已保存 magnon_band.png")
print(f"Γ 能隙 = {w[0,0]:.3f} meV (2S|A| = {2*1.5*0.5:.3f})")
print(f"K 点: {w[nseg,0]:.2f} / {w[nseg,1]:.2f} meV")
print(f"M 点: {w[2*nseg,0]:.2f} / {w[2*nseg,1]:.2f} meV")
