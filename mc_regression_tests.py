#!/usr/bin/env python3
"""Monte_Carlo_skyrmions_improved 回归测试（2026-08-14 三方审核修复后固化）。

覆盖三方审查确认的 CRITICAL/MAJOR 修复点：
  C3  local_field 跨子格 0.5 因子（Nb=2 动力学场）
  C4  ΔS_M μ_s/k_B 因子（J=0 解析对照）
  C5  spin_structure_factor 频域相位 + 子格相干（honeycomb FM S(0)=N）
  M1  run_phase_diagram Nb≠1 不崩溃
  M2  export_xyz species 列表格式
  M3  PT 接受率 ∈ [0,1]（分母修复）
  + 核心引擎回归（能量精确、Q(FM)=0、DMI 约定）

运行：python3 mc_regression_tests.py   （无依赖，纯 assert）
"""
import sys
import numpy as np

from mc import (Lattice, Hamiltonian, MonteCarlo2D,
                KB_MEV_PER_K, MU_S_MEV_PER_T)

PASS = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  ✓ {name}")


def tri_lattice(Nx=4, Ny=4, J=-10.0, D=1.45, A=-0.04):
    """三角格 FM + DMI（与 run.py 一致）。"""
    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[0.0, 0.0]], Nx, Ny)
    ham = Hamiltonian(Nb=1, A_ani=A)
    for off, dd in [([1, 0], D), ([0, 1], D), ([-1, 1], D)]:
        Jm = np.array([[J, 0.0, -dd], [0.0, J, 0.0], [dd, 0.0, J]])
        ham.add_bond(0, 0, off, Jm)
    return lat, ham


def test_core_engine():
    print("[core] 引擎回归")
    lat, ham = tri_lattice()
    mc = MonteCarlo2D(lat, ham, seed=1)
    mc.spins[:] = 0.0
    mc.spins[..., 2] = 1.0                      # FM +z
    E = mc.total_energy() / mc.lat.N_total
    assert abs(E - (-30.04)) < 1e-6, f"FM 能量 {E} != -30.04"
    ok(f"FM+z 能量 = {E:.4f} meV/spin (期望 -30.04)")
    assert abs(mc.topological_charge()) < 1e-9
    ok("Q(FM) = 0")
    # DMI 约定：bond 沿 x̂ → d = D·ŷ（矩阵 A_02 = −D）
    A_mat = ham.bonds[0][0][3]                   # 第一条 (0,0,[1,0],J)
    A_asym = (A_mat - A_mat.T) / 2
    assert abs(A_asym[0, 2] + 1.45) < 1e-12 and abs(A_asym[1, 0]) < 1e-12
    ok("DMI d = D·(ẑ×r̂) 约定（bond[1,0] → A_02 = −D）")


def test_local_field_nb2():
    print("[C3] Nb=2 有效场（跨子格 0.5 修复）")
    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 2, 2)
    ham = Hamiltonian(Nb=2, A_ani=0.0)
    ham.add_bond(0, 1, [0, 0], np.diag([-6.0] * 3))
    mc = MonteCarlo2D(lat, ham, seed=1)
    mc.spins[:] = 0.0
    mc.spins[..., 2] = 1.0
    # 数值 −∂H/∂S（有限差分）对照动力学 1 步旋转角
    d = 1e-6
    S0 = mc.spins[0, 0, 0].copy()
    E0 = mc.total_energy()
    grad = np.zeros(3)
    for c in range(3):
        mc.spins[0, 0, 0, c] += d
        Ep = mc.total_energy()
        mc.spins[0, 0, 0, c] -= 2 * d
        Em = mc.total_energy()
        grad[c] = (Ep - Em) / (2 * d)
        mc.spins[0, 0, 0, c] = S0[c]
    true_h = -grad
    # 动力学一步：S⊥h 时旋转速率 = |h|
    mc.spins[0, 0, 0, 0] = 1e-3                 # 轻微倾斜 ⊥ h（进动不改变 S_z）
    mc.spins[0, 0, 0, 2] = np.sqrt(1 - 1e-6)
    traj, _ = mc.run_spin_dynamics(dt=0.01, n_steps=1, T=0.0, damping=0.0, save_interval=1, seed=1)
    phi0 = np.arctan2(1e-3, np.sqrt(1 - 1e-6))
    phi1 = np.arctan2(traj[0, 0, 0, 0, 1], traj[0, 0, 0, 0, 0])
    ang = (phi1 - phi0) % (2 * np.pi)           # xy 面进动角 = dt·|h|
    assert abs(true_h[2] - 6.0) < 1e-6, f"true -dH/dS_z = {true_h[2]} != 6"
    assert abs(ang - 0.01 * 6.0) < 0.01 * 6.0 * 0.1, f"进动角 {ang} != dt·|h|=0.06"
    ok(f"跨子格有效场 |h| = {true_h[2]:.1f} meV（修复前为 3.0）; 进动角 {ang:.4f} ≈ dt·|h|")


def test_mce_mu_s():
    print("[C4] ΔS_M μ_s/k_B 因子（J=0 解析对照）")
    lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], 16, 16)
    ham = Hamiltonian(Nb=1, A_ani=0.0)          # J=0：独立自旋顺磁
    mc = MonteCarlo2D(lat, ham, seed=7)
    T_list = np.linspace(60.0, 4.0, 12)
    B_list = [0.0, 3.0, 6.0]
    T, Bs, S_abs, dS_M, C, dT_ad = mc.run_magnetocaloric(
        T_list=T_list, B_list=B_list, equip_steps=2000, calc_steps=3000,
        sample_interval=2, output_file=None)
    for Ti in (2, 6, 10):                        # T ≈ 50, 30, 10 K
        T0 = T_list[Ti]
        x = MU_S_MEV_PER_T * 6.0 / (KB_MEV_PER_K * T0)
        ana = np.log(np.sinh(x) / x) - x / np.tanh(x) + 1.0
        assert dS_M[Ti, -1] < 0.0, "ΔS_M 符号（加场熵减）"
        rel = abs(dS_M[Ti, -1] / ana - 1.0)
        # 容差：相对 25% 与绝对 0.02 k_B 取大者（高 T 信号弱，绝对误差主导）
        assert rel < 0.25 or abs(dS_M[Ti, -1] - ana) < 0.02, \
            f"T={T0}: dS_M={dS_M[Ti,-1]:.4f} vs analytic {ana:.4f} (rel {rel:.2f})"
        ok(f"T={T0:5.1f}K: dS_M={dS_M[Ti,-1]:.4f} k_B/spin vs 解析 {ana:.4f} (rel err {rel:.2f})")
    assert abs(mc.ham.B_field_meV).sum() == 0.0, "MCE 后外场未恢复"
    ok("MCE 后 B_field 恢复")


def test_sqf_honeycomb():
    print("[C5] honeycomb S(q) 子格相干（FM S(0) = N）")
    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 4, 4)
    ham = Hamiltonian(Nb=2, A_ani=0.0)
    for off in ([0, 0], [-1, 0], [0, -1]):
        ham.add_bond(0, 1, off, np.diag([-6.0] * 3))
    mc = MonteCarlo2D(lat, ham, seed=1)
    mc.spins[:] = 0.0
    mc.spins[..., 2] = 1.0
    q1, q2, S = mc.spin_structure_factor()
    i0 = int(np.argmin(np.abs(q1))); j0 = int(np.argmin(np.abs(q2)))
    N = mc.lat.N_total
    assert abs(S[i0, j0] - N) < 1e-6, f"S(0) = {S[i0, j0]} != N = {N}"
    ok(f"S(q=0) = {S[i0, j0]:.1f} = N（修复前非相干求和 = N/2）")


def test_phase_diagram_nb2():
    print("[M1] run_phase_diagram Nb=2 不崩溃")
    lat = Lattice([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], 2, 2)
    ham = Hamiltonian(Nb=2, A_ani=-0.5)
    for off in ([0, 0], [-1, 0], [0, -1]):
        ham.add_bond(0, 1, off, np.diag([-6.0] * 3))
    mc = MonteCarlo2D(lat, ham, seed=1)
    res = mc.run_phase_diagram(B_list=[0.0, 1.0], T_list=[20.0, 5.0],
                               equip_steps=2, calc_steps=4, sample_interval=1,
                               classify=True, verbose=False)
    assert res["phases"].shape == (2, 2)
    ok("Nb=2 相图扫描完成（Q/n_sk 跳过，不抛 NotImplementedError）")


def test_export_xyz_species():
    print("[M2] export_xyz species 列表")
    lat, ham = tri_lattice(2, 2)
    mc = MonteCarlo2D(lat, ham, seed=1)
    text = mc.export_xyz("/tmp/rt_spins.xyz", species=["Cr"] * 4, spin_scale=1.0)
    first_atom = text.split("\n")[2].split()[0]
    assert first_atom == "Cr", f"首原子物种 = {first_atom!r}（修复前为 Python 列表 repr）"
    ok("species 列表逐原子写入")


def test_pt_acc_rate():
    print("[M3] PT 接受率分母（报告值 ∈ [0,1]）")
    lat, ham = tri_lattice(4, 4)
    mc = MonteCarlo2D(lat, ham, seed=1)
    res = mc.run_parallel_tempering(T_list=[5.0, 15.0, 40.0], equip_steps=5,
                                    swap_interval=3, n_swaps=4,
                                    B_field=(0.0, 0.0, 1.0), seed=1, verbose=False)
    acc = np.asarray(res["acc_rate"])
    assert acc.shape == (2,) and np.all(acc >= 0.0) and np.all(acc <= 1.0 + 1e-9), acc
    ok(f"acc_rate = {acc}（分母 n_swaps，修复前报告值减半）")


if __name__ == "__main__":
    test_core_engine()
    test_local_field_nb2()
    test_mce_mu_s()
    test_sqf_honeycomb()
    test_phase_diagram_nb2()
    test_export_xyz_species()
    test_pt_acc_rate()
    print(f"\n全部 {PASS} 项回归测试通过 ✅")
