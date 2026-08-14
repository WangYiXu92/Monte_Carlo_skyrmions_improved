# Monte_Carlo_skyrmions (improved)

通用的二维经典自旋 Monte Carlo 程序：Skyrmion / 居里温度 / 磁滞回线。

原始 repo: https://github.com/turney0524/Monte_Carlo_skyrmions/
本 fork 修正了例子的物理错误并补上工程化能力（详见 CHANGELOG 段）。

## Features

1. **Skyrmion**：固定磁场下模拟退火，输出自旋构型 + 拓扑数 Q
2. **Curie temperature**：温度扫描输出平均磁矩 M(T)、磁化率 χ(T)、热容 C(T)
3. **Hysteresis loop**：磁场扫描输出 <M_z>(B)，保留历史 → 真磁滞
4. **Magnetocaloric (MCE)**：多场温度扫描 → ΔS_M(T)（麦克斯韦关系）+ ΔT_ad(T)（绝热温变）
5. **Spin structure factor S(q)**：自旋构型 FFT → 识别磁序（FM q=0 峰 / 螺旋 q* / skyrmion 晶格 Bragg 峰）
6. **Magnon spectra**：共线 FM 线性自旋波（HP 一阶）色散 ω(k)（多子格、各向异性隙、Zeeman 隙）
7. **Magnetic structure analysis**：自动识别磁序（FM/AFM/helical/skyrmion/PM）+ 提取磁性单胞（q* → 磁平移 → 商空间）

## 模型

```
H = Σ_<ij> S_i J_ij S_j + Σ_i A_i (S_iz)² − Σ_i μ_s B·S_i
```

- J_ij 是 3×3 矩阵：对称部分 = 各向同性交换，反对称部分 = DMI
- **DMI 约定：轴向矢量 d = D·(ẑ×r̂)**（文献标准界面型，D>0 = 正手性 Néel skyrmion）
  （已验证三个例子的矩阵均满足此约定——root/MX2/AFM_honeycomb 原版即正确）
- J < 0 铁磁，J > 0 反铁磁；A < 0 易 z 轴
- 单位：能量 meV，温度 K（k_B = 0.08617333262 meV/K），磁场 Tesla（μ_s = gμ_B·S = 0.1157676 meV/T, g=2, S=1）
- 任意 2D 晶格：输入晶格基矢 + 分数坐标 basis（多原子支持）+ bond 列表 [i, j, offset]

## 用法

```bash
# 每个例子目录自包含（有自己的 mc.py），直接运行：
python run.py            # 根目录：三角格 skyrmion 退火
cd CrI3_test_curie && python run.py     # honeycomb 双子格居里温度
cd MX2 && python run.py                 # 三角格磁滞回线
cd AFM_honeycomb && python run.py       # AFM honeycomb + DMI skyrmion
cd MCE_demo && python run.py            # CrI3 型磁热效应 ΔS_M + ΔT_ad
```

改 `SIMULATION_MODE` 和对应参数块即可切换功能。

### 磁热效应（MCE）

```python
T, B, S_abs, dS_M, C, dT_ad = mc.run_magnetocaloric(
    T_list=np.linspace(100, 1, 30), B_list=[0, 2, 4, 6],
    equip_steps=3000, calc_steps=4000, sample_interval=2,
)
```

- **ΔS_M 用麦克斯韦关系**：ΔS_M(T,ΔB) = (μ_s/k_B)·∫₀^{ΔB} (∂M/∂T)_{B'} dB'（M 为每自旋无量纲平均、B 用 Tesla；μ_s/k_B≈1.343 不可省；T→0 自动归零）
- **ΔT_ad 用等熵构造（严格）**：S(T₂,B) = S(T₁,0) → ΔT_ad = T₂−T₁（反插值；不受 C 噪声影响，无解为 NaN）
- **单位**：`molar_mass_g_per_mol` 给定时 ΔS_M 与 C 输出 **J/(kg·K)**（换算 R/M×10³；CrI₃=432.7 → |ΔS_M| 峰 3.09 J/(kg·K) @ 35 K），None 时 k_B/自旋
- 绝对熵 S(T,B) = ln(4π) − ∫₀^β (E−E₀)dβ' + β(E−E₀) 也输出（低 T 需充分平衡否则不可靠）
- C(T) = N·Var(e)/(k_B T²)（涨落公式）
- 验收：ΔS_M<0（常规 MCE）、|ΔS_M| 峰在 Tc、ΔT_ad>0、T→0 归零
- 已验证：J=0 体系绝对熵与单自旋解析解吻合；CrI3 演示 ΔT_ad 峰 6.4 K @ 39 K（⚠️ 2026-08-14 修正 μ_s/k_B 因子后 |ΔS_M| 数值待重跑 demo 更新）

### 自旋结构因子

```python
q1, q2, S = mc.spin_structure_factor()   # FFT 频域索引 + S(q) 网格
```

### 磁振子色散（线性自旋波）

```python
kpath = [(0,0), (1/3,1/3), (0.5,0), (0,0)]   # 分数坐标
ks, omega = mc.magnon_spectrum(kpath, S=1.5, B_field=(0,0,0))  # meV
```

- **公式**：ω_k = S|J|Σ_δ(1−cos k·δ) + 2S|A| + μ_s·B_z（多子格取 bond 矩阵 J_zz 与横向分量；A<0 易轴给 2S|A| 隙）
- **已校准**：1D 4-环 S=1..3 精确对角化——k=π/2 精确匹配；k=π 处 0.75/0.875/0.917 = 1/(2S) 量子修正收敛（LSWT 是经典极限）
- **限制**：仅共线 FM；DMI（反对称 J）忽略（基态倾斜，需非共线 LSWT）；AFM/非共线不支持
- 示例：`cd MCE_demo && python3 magnon_demo.py` → magnon_band.png（honeycomb CrI3 双带：Γ 隙 1.5 meV = 2S|A|）

### 磁结构自动识别 + 磁单胞提取

```python
r = mc.magnetic_structure_analysis()
# r['order']       : 'FM' | 'AFM' | 'helical' | 'skyrmion_lattice' | 'multi-q' | 'PM'
# r['q_stars']     : 磁传播矢量 q*（S(q) 峰，分数坐标）
# r['top_charge']  : 拓扑荷 Q（skyrmion 检测）
# r['cell_matrix'] : 磁单胞基矢（超胞格点坐标）
# r['cell_spins']  : 单胞内自旋构型 (N1, N2, Nb, 3)
# r['cell_repeats']: 超胞内含磁单胞数
```

- **原理**：S(q) 峰 → q*（磁传播矢量）；磁单胞 = 满足 q*·R ∈ ℤ 的所有磁平移 R 的等价类代表元（商空间）
- **验证**：五类构型全部通过——FM（1×1）、checkerboard AFM（2 格点 [+1,−1]）、螺旋 q=(1/4,0)（4 态 [1,0,−1,0]）、skyrmion 晶格（12×12 单胞, Q=±16 精确）、随机 → PM
- **端到端**：真实 MC 退火（48×48, B=0.15T）出 Q=−1 skyrmion 自动识别 ✓
- 示例：`cd MCE_demo && python3 analyze_magnetic_demo.py`
- 注意：单个（非晶格）skyrmion 无晶格周期 → 无 q* 峰、单胞=整胞（Q 仍是判别键）

**导出 VASP POSCAR（磁结构 → DFT 输入）**：

```python
text, magmom = mc.export_magnetic_cell_poscar("magnetic_cell.POSCAR",
                                              spin_scale=1.0, species=["Cr"])
# 生成 magnetic_cell.POSCAR（2D 晶格 + z 真空层）+ magnetic_cell.POSCAR.magmom
# MAGMOM = Sx Sy Sz ...（单位矢量 × spin_scale），直接粘贴进 INCAR
```

- 晶格矢量 = 磁单胞基矢 × 原胞 a_vecs；坐标 = 单胞格点笛卡尔 → 分数（wrap）
- 已验证：AFM（2 原子 ±z）、skyrmion 晶格（144 原子 = 12×12 单胞，|m|=2.5 处处，单胞面积精确）

### Skyrmion 定位与统计（热稳定性）

```python
centers = mc.skyrmion_positions()          # [(x, y, Q_local), ...]：mz 局部极小 + 拓扑荷验证
st = mc.skyrmion_statistics()              # 半径（mz=0 等值面）、密度、晶格常数、total_Q
Ts, Ns, Nstd = mc.skyrmion_stability(T_list, equip_steps, calc_steps)  # 熔化曲线
```

- **定位**：mz 局部极小（< −0.5）→ 周围 r=2 内 |Σρ_Q| ≥ 0.25 验证（ρ_Q 为 Berg–Lüscher 三角形 1/3 分摊）→ 邻近去重。已验证：12×12 晶格 16/16 精确、单 skyrmion 1/1
- **统计**：半径沿 6 个 NN 方向 mz 过零插值；晶格常数 = 中心间最小周期距离
- **稳定性**：升温扫描逐 T 平衡 + 统计 skyrmion 数 → 阶跃熔化曲线（demo 实测：J=−40/D=4/B=1T 下 9 个 skyrmion 3-6K 全活 → 12K 半数 → 30K 殆尽 → 36K 归零）
- ⚠️ 物理窗口：D/J 太弱（<0.06）时 skyrmion 晶格不是哈密顿量稳定态，构造晶格一加热即湮灭（小 λ 势垒低）；稳定曲线需 D/J ≳ 0.1 + 合适 B
- 示例：`cd MCE_demo && python3 skyrmion_analysis_demo.py` → skyrmion_stability.png

### Langevin 自旋动力学 + S(q,ω) 动态结构因子（INS 对接）

```python
traj, times = mc.run_spin_dynamics(dt=0.002, n_steps=40000, T=0.3,
                                   damping=0.05, save_interval=10)   # Heun 积分
qs, omega, S = mc.dynamic_structure_factor(traj, times, q_grid=[(4,0), ...])
```

- **LLG + 热噪声（Langevin）**：∂S/∂t = −S×H_eff − λS×(S×H_eff) + 噪声（<ξξ'> = 2λk_BTδ），Heun 二阶积分
- **验证**：① 无耗散 λ=0/T=0 能量守恒 0.0003%（2000 步）；② **S(q,ω) 峰位 vs LSWT 理论 0.97-1.08**（12×12 FM 四 q 点）；demo 24×24 平均偏差 6%
- **S(q,ω) = ∫dt e^{iωt}⟨S_q(t)·S_{−q}(0)⟩**——直接对应非弹性中子散射（INS）截面
- ⚠️ 稳定性要求：dt·ω_max ≲ 0.15（ω_max = S·z·|J| 量级）；Heun 需 `S0.copy()` 防引用污染（历史爆炸根因）
- 示例：`cd MCE_demo && python3 spin_dynamics_demo.py` → sw_spectrum.png（S(q,ω) 强度图 + LSWT 色散线叠加）

### OVITO/XYZ 导出（磁构型动画）

```python
mc.export_xyz("spins.xyz", species="Cr", spin_scale=1.0)            # 单帧
mc.export_xyz("traj.xyz", trajectory=traj, spin_scale=1.0)          # 动力学轨迹多帧
```

- 标准 .xyz 多帧格式 + Lattice 头（OVITO 直接打开渲染磁矩矢量场）
- 每原子 3 列 = 自旋矢量 × spin_scale（OVITO Vector 属性）
- 已验证：单帧 36 原子 |spin|=2.0 处处、多帧 10 帧结构正确

### B-T 磁相图扫描（B–T phase diagram）

```python
res = mc.run_phase_diagram(
    B_list=[0.0, 0.5, 1.0, 2.0], T_list=[5, 20, 40, 80, 150, 200],
    equip_steps=500, calc_steps=250, sample_interval=10,
    protocol="cooling", classify=True, output_file="phase.csv")
# res['phases'] : (len(B), len(T)) 相标签（FM/AFM/helical/skyrmion_lattice/multi-q/PM）
# res['M'], res['Q'], res['n_sk'] : 同形状数值数组
```

- **协议三选一**：`cooling`（field-cooled，逐 T 降温 warm start——skyrmion 口袋最易出现）、`heating`（ZFC-like，滞后对照）、`fresh`（每点独立随机态，无记忆）
- 相标签 = `magnetic_structure_analysis` 分类，skyrmion 计数（n_sk≥1 且 Q≠0）优先覆盖 S(q) 标签
- ⚠️ 滞后警告：相边界依赖冷却/加热路径（亚稳态卡滞是真实物理）；生产扫描请加长 equip_steps 或配合 Parallel Tempering
- 示例：`PhaseDiagram_demo/phase_diagram_J10.csv/png`（J=−10/D=1.45/A=0.04，24×24，6×11 网格，B=1T 低温 skyrmion 口袋可见）

### Skyrmion 扩散 + 寿命（动力学可观测量）

```python
traj, times = mc.run_spin_dynamics(dt=..., n_steps=..., T=..., damping=0.05)
res = mc.skyrmion_diffusion(traj, times)     # 质心轨迹 → MSD → D（Einstein 4Dt）
t, N, N0, tau, R2 = mc.skyrmion_lifetime(traj, times)   # N(t) 指数衰减 → τ
eb = mc.arrhenius_analysis(T_list, tau_list) # ln τ vs 1/T → 湮灭势垒 E_b (meV)
```

- **扩散**：逐帧 `skyrmion_positions` → PBC 最小镜像配对 → MSD(τ) → 前 1/3 线性段 Einstein D
- **寿命**：N(t) log 线性拟合（非指数段自动拒绝）；**Arrhenius**：τ(T)=τ₀exp(E_b/k_BT)
- ⚠️ 需要**真实 skyrmion 构型**作起点（MC 退火生成或实验构型）；三角格解析 ansatz 离散化后局部 Q 不足（~0.3），会检测碎片化——合成轨迹验证 MSD 数学链可用（demo 见 `SkyrmionDynamics_demo/`）

### Parallel Tempering（副本交换 MC）

```python
res = mc.run_parallel_tempering(
    T_list=[3, 6, 12, 24, 48, 96], equip_steps=80, swap_interval=30,
    n_swaps=15, B_field=(0,0,1), seed=42)
# res['spins_final'] : 各副本最终构型；res['E_hist'] : 最低温副本能量轨迹
# res['acc_rate']    : 相邻副本对交换接受率
```

- 标准副本交换：本地 Metropolis + 相邻温度副本交换（Metropolis 判据），奇偶轮交替；walker 温度跟踪；交换决策用独立 RNG（可复现）
- **用途**：克服亚稳态/势垒卡滞（如 skyrmion 相退火失败时）；副本温度间距建议几何分布、目标接受率 10–30%（过宽 → 接受率≈0 退化为并行独立链）
- 验证：16×16 随机起点 6 副本 15 轮 → 低温副本能量 −22700 → −30655（基态），接受率 0.10/0.13/0.10

### S(q,ω) 峰位/线宽提取（磁振子色散 + 阻尼）

```python
qs, omega, S = mc.dynamic_structure_factor(traj, times, q_grid=[(1,0),(2,0),(3,0)])
res = mc.sqw_peak_extraction(S, omega, q_grid, fit="lorentzian")
# res['omega_peak'] : 峰位 ω(q)；res['FWHM'] : 线宽；res['R2'] : 拟合优度
```

- Lorentzian（阻尼磁振子线型，FWHM=2Γ）或 Gaussian 拟合；基线 = 最低 20% 分位中位数
- **验证**：12×12 FM J=−10 与 LSWT 对照 ν=|J|Σ₃(1−cos 2πq·δ)/(2π)，五 q 点比值 0.91–0.99（与动力学验证 0.97–1.08 同源）
- ⚠️ 单位：rfftfreq 给出**循环频率** ν（= ω_meV/2π）；Σ 取 3 个正 NN 方向（± 对称）

### 多 seed 并行（统计误差估计）

```python
from mc import run_curie_temperature_seeds
mean, std, all_res = run_curie_temperature_seeds(
    ham_spec={"A_ani": -0.5, "bonds": [(0,1,[0,0],J), (0,1,[-1,0],J), (0,1,[0,-1],J)]},
    a_vecs=..., basis=..., Nx=60, Ny=60,
    T_list=np.linspace(1, 100, 40), equip_steps=2000, calc_steps=3000,
    n_seeds=4, n_workers=4, base_seed=0,
)
```

## 相对原版的改进（CHANGELOG）

1. **CrI3 例子修正**：原版是单原子三角格（4 配位 + 多余 on-site 自键），现改为真 honeycomb 双子格（basis 2 原子、3 条 A→B 最近邻 bond）
2. **MX2 例子修正**：原版与根目录逐字节重复，现改为磁滞回线演示（三角格 CrX2 型有效模型），三个例子 = 三种模式各一
3. **随机种子**：`MonteCarlo2D(..., seed=...)`，所有 run.py 带 `SEED` 常量 → 可复现
4. **接受率自适应**：`proposal_angle=None` 时自动调角（目标接受率 30–60%），低 T 弛豫不再依赖手调
5. **多 seed 并行**：`run_curie_temperature_seeds()` 多进程并行，输出 M/χ 的 mean±std（统计误差）
6. **拓扑数注释**：明确 Berg–Lüscher NN 三角形剖分（三角格上菱形劈两个 NN 三角形，原实现即正确）
7. **AFM_honeycomb 注释修正**：J=+10 是反铁磁（原文注释误写"铁磁"）
8. **DMI 约定确认**：三例子矩阵均满足 d = D·(ẑ×r̂)（文献约定），无需修改。⚠️ 物理观察：90×90 大格退火容易陷进负手性多 skyrmion 亚稳态（实测 Q=−2/−3/−4，B=1T 场能 0.116 meV 太弱不足以选择涡旋方向 → 手性近简并）；30×30 小格稳定出 Q=+1。模拟 skyrmion 晶格建议用更强的 B 或 field-cooling
9. **CrI3 例子平衡修复**：heating 扫描低 T 从随机初态弛豫极慢（M(1K)≈0.2 伪值）→ 改 cooling 扫描（100→1K）+ 30×30 + 5000 sweeps/T（实测 Tc≈37 K）
10. **MX2 磁滞物理修复**：原版 J=−8/A=−0.05 场翻不动且太软（无磁滞）→ J=−0.5/A=−1.5/T=2K（Ising 型成核翻转，实测矫顽场 ~15 T）
11. **skyrmion 退火加长**：steps_per_T 2000→4000（90×90 淬火过快冻结亚稳态）
12. **2026-08-14 三方审核修复批次**（AGY/Codex/Claude + 数值实证，回归测试 `mc_regression_tests.py` 12 项固化）：
    - Langevin 噪声加 k_B 因子（√(2λk_BT/dt)，原幅度大 3.4×）；Heun 校正步复用同一噪声增量（原重新抽样破坏涨落-耗散）
    - `local_field` 跨子格键 0.5 因子修正（Nb>1 有效场曾砍半；on-site 键同步修正）
    - ΔS_M 加 μ_s/k_B≈1.343 因子（原小 0.744×，J=0 解析对照验证）；MCE 的 M 改逐采样平均并用 M_z（原单次快照 + |M| 高 T 偏置）
    - S(q) 子格相位移至频域 + 子格相干求和（honeycomb FM S(0)=N 验证）；S(q,ω) 零填充线性相关 + 单边因子 2 + 全子格
    - B-T 相图：Nb≠1 不再崩溃（Q/n_sk 跳过）、协议显式排序（cooling=降温，输入 T 顺序任意）、采样计数修正、B 场 try/finally 恢复
    - PT 改固定温度槽表述（修复混合 walker/slot 导致的细致平衡破坏）+ 接受率分母修正（原报告值减半）
    - E₀ 外推改 T 线性拟合（经典 equipartition；T² 拟合实测高估 +0.4~0.8 meV）
    - POSCAR 多物种按物种分组逐物种计数；export_xyz species 列表逐原子写入；MSD unwrap 去最小镜像截断；seeds 输出 18 列头；skyrmion 中心 PBC 最小镜像去重
    - ⚠️ MCE/动力学输出数值因此修正而变化：README 旧数值（|ΔS_M| 3.09 J/kg/K 等）以重跑 demo 为准

## 已知限制

- 经典自旋（|S|=1），无量子涨落；Tc 为经典估计
- 单线程内核（numba njit）；并行只在多 seed 层
- χ 的误差来自多 seed 散布，单 seed 内无 binning（自相关未完全扣除）
- 拓扑数仅限 Nb=1 三角格
