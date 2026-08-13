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

- **ΔS_M 用麦克斯韦关系**：ΔS_M(T,ΔB) = ∫₀^{ΔB} (∂M/∂T)_{B'} dB'（稳、无外推、T→0 自动归零）
- 绝对熵 S(T,B) = ln(4π) − ∫₀^β (E−E₀)dβ' + β(E−E₀) 也输出（低 T 需充分平衡否则不可靠）
- C(T) = N·Var(e)/(k_B T²)（涨落公式）
- 验收：ΔS_M<0（常规 MCE）、|ΔS_M| 峰在 Tc、ΔT_ad>0、T→0 归零
- 已验证：J=0 体系与单自旋解析解吻合（±0.006 k_B）；CrI3 演示 |ΔS_M| 峰 0.16 k_B/spin @ 35 K ≈ Tc

### 自旋结构因子

```python
q1, q2, S = mc.spin_structure_factor()   # FFT 频域索引 + S(q) 网格
```

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

## 已知限制

- 经典自旋（|S|=1），无量子涨落；Tc 为经典估计
- 单线程内核（numba njit）；并行只在多 seed 层
- χ 的误差来自多 seed 散布，单 seed 内无 binning（自相关未完全扣除）
- 拓扑数仅限 Nb=1 三角格
