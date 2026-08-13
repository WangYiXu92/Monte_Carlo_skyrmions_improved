"""磁热效应（MCE）演示：CrI3 型 honeycomb 双子格。

计算 ΔS_M(T, ΔB)（磁熵变）与 ΔT_ad(T)（绝热温变）：
  S(T,B) = N·ln(4π) − ∫₀^{β} U(β',B) dβ'     热力学积分（β = 1/k_B T）
  ΔS_M   = S(T,B₂) − S(T,B₁)                   常规 MCE: < 0（加场熵减）
  ΔT_ad  = −T·ΔS_M/C(T)                        绝热磁化升温: > 0
  C(T)   = Var(E)/(k_B T²)                      涨落公式

验收判据：
  1. ΔS_M < 0（常规 MCE），|ΔS_M| 峰值在 Tc ≈ 37 K 附近
  2. T → 0 时 ΔS_M → 0（第三定律）
  3. ΔT_ad > 0，峰值也在 Tc 附近
"""
import numpy as np

from mc import Lattice, Hamiltonian, MonteCarlo2D

SEED = 20260813

# =====================================================================
# 1. 晶格参数：honeycomb（同 CrI3_test_curie）
# =====================================================================
a = 1.0
a_vecs = [[a, 0.0], [0.5 * a, np.sqrt(3.0) / 2.0 * a]]
basis = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
Nx, Ny = 30, 30

# =====================================================================
# 2. 哈密顿量参数（CrI3 型：J = -6 meV FM, A = -0.5 meV 易 z 轴）
# =====================================================================
A_ani = -0.5
J_ex = -6.0


def build_simulator():
    lattice = Lattice(a_vecs, basis, Nx, Ny)
    ham = Hamiltonian(Nb=len(basis), A_ani=A_ani)
    for off in ([0, 0], [-1, 0], [0, -1]):
        J_mat = np.diag([J_ex] * 3)
        ham.add_bond(0, 1, off, J_mat)
    return MonteCarlo2D(lattice, ham, seed=SEED)


# =====================================================================
# 3. MCE 参数：温度扫描（cooling）+ 磁场循环
# =====================================================================
MCE_PARAMS = {
    "T_list": np.linspace(100.0, 1.0, 30),   # K（降温方向）
    "B_list": [0.0, 2.0, 4.0, 6.0],          # Tesla
    "equip_steps": 3000,
    "calc_steps": 4000,
    "sample_interval": 2,
    "output_file": "mce_results.txt",
}

if __name__ == "__main__":
    mc = build_simulator()
    T, B, S, dS_M, C, dT_ad = mc.run_magnetocaloric(**MCE_PARAMS)

    # 摘要输出
    print("\n===== MCE 摘要 =====")
    print("T(K)   ΔS_M(0→6T) [kB/spin]   ΔT_ad(0→6T) [K]   C [kB/spin]")
    i6 = len(B) - 1
    for i in range(0, len(T), 3):
        print(f"{T[i]:5.1f}   {dS_M[i, i6-1]:+10.4f}          {dT_ad[i, i6-1]:+8.2f}        {C[i, 0]:7.4f}")
    i_pk = np.argmin(dS_M[:, i6-1])
    print(f"\n|ΔS_M| 峰值: T = {T[i_pk]:.1f} K, ΔS_M = {dS_M[i_pk, i6-1]:.4f} kB/spin (0→6 T)")
    if np.all(np.isnan(dT_ad[:, i6-1])):
        print("ΔT_ad: 无等熵解（绝对熵曲线未覆盖所需范围）")
    else:
        ipk2 = np.nanargmax(dT_ad[:, i6-1])
        print(f"ΔT_ad 峰值(等熵法): T = {T[ipk2]:.1f} K, ΔT_ad = {dT_ad[ipk2, i6-1]:.2f} K")
