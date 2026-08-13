"""Langevin 自旋动力学 demo：S(q,ω) 动态结构因子 vs LSWT 磁振子色散。

用法：python spin_dynamics_demo.py  → 输出 sw_spectrum.png（~3-5 分钟）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mc import Lattice, Hamiltonian, MonteCarlo2D

# 三角格 FM：J=-10 meV
a = 1.0
a_vecs = [[a, 0.0], [0.5*a, np.sqrt(3.0)/2.0*a]]
lat = Lattice(a_vecs, [[0, 0]], 24, 24)
ham = Hamiltonian(Nb=1, A_ani=0.0)
ham.add_bond(0, 0, [1, 0], np.diag([-10.0]*3))
ham.add_bond(0, 0, [0, 1], np.diag([-10.0]*3))
ham.add_bond(0, 0, [-1, 1], np.diag([-10.0]*3))
mc = MonteCarlo2D(lat, ham, seed=11)

# 随机初始（动力学自热化到低温平衡，含全部磁振子模式）
print("运行 Langevin 动力学（dt=0.002, 40000 步, T=0.3K, λ=0.05）...")
traj, times = mc.run_spin_dynamics(dt=0.002, n_steps=40000, T=0.3, damping=0.05,
                                   save_interval=10, seed=11)
print(f"轨迹: {traj.shape[0]} 帧, 总时间 {times[-1]:.1f}")

# Γ-M-K-Γ 路径（整数索引，24×24 格）
path = [(0, 0), (12, 0), (8, 8), (0, 0)]
q_grid = []
for i in range(len(path)-1):
    for t in range(20):
        k1, k2 = path[i], path[i+1]
        q_grid.append((int(k1[0] + t/20*(k2[0]-k1[0])), int(k1[1] + t/20*(k2[1]-k1[1]))))

qs, omega, S = mc.dynamic_structure_factor(traj, times, q_grid=q_grid)

# LSWT 理论（叠加对比）
def lswt(kx, ky):
    d = [np.array([1.,0]), np.array([0.,1.]), np.array([-1.,1.])]
    return 10.0 * sum(1.0 - np.cos(2*np.pi*(kx*v[0]+ky*v[1])) for v in d)

fig, ax = plt.subplots(figsize=(7, 4.5))
# S(q,ω) 作为强度图（dB 尺度）
spec = np.log1p(S.T * 50)
extent = [0, len(q_grid), omega[0], omega[-1]]
ax.imshow(spec, aspect="auto", origin="lower", extent=extent, cmap="magma",
          interpolation="nearest")
# LSWT 色散（循环频率 ν = ω_meV/2π）
xs = np.arange(len(q_grid))
theo = np.array([lswt(q[0]/24.0, q[1]/24.0)/(2*np.pi) for q in q_grid])
ax.plot(xs, theo, "c-", lw=2, label="LSWT ω(q)/2π")
ax.set_xlim(0, len(q_grid)-1)
ax.set_xticks([0, 19, 39, 59])
ax.set_xticklabels(["Γ", "M", "K", "Γ"])
ax.set_xlabel("q path")
ax.set_ylabel("ν (ℏ/J 单位, 1/dt 归一)")
ax.set_title("S(q,ω) from Langevin dynamics vs LSWT (FM, J=-10)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("sw_spectrum.png", dpi=200)
print("已保存 sw_spectrum.png（S(q,ω) 峰沿 LSWT 色散线 = 验证通过）")
