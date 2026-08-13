# mc.py
import numpy as np
from numba import njit

# ==========================================
# 物理常数定义区 (用于单位转换)
# ==========================================
KB_MEV_PER_K = 0.08617333262  # 玻尔兹曼常数 (meV/K)
MU_S_MEV_PER_T = 0.1157676    # 有效磁矩 (meV/T)，假设 g=2, S=1 (可根据具体材料调整)

# 接受率自适应控制参数
_ACCEPT_TARGET_LO = 0.30
_ACCEPT_TARGET_HI = 0.60
_ANGLE_ADJUST = 1.05
_ANGLE_MIN = 0.01
_ANGLE_MAX = 0.50
_ADAPT_EVERY = 10          # 每 N 个 sweep 调整一次提案角

# ==========================================
# 核心 Numba 加速算法区
# ==========================================
@njit(cache=True)
def _local_energy_numba(x, y, b, S_val, spins, Nx, Ny, num_bonds, bond_targets, bond_matrices, A_ani, B_field_meV):
    """Numba 加速的单自旋局域能量计算 (所有能量单位均为 meV)"""
    E = 0.0

    # 1. 交换耦合与 DMI 相互作用
    n_bonds = num_bonds[b]
    for i in range(n_bonds):
        neighbor_b = bond_targets[b, i, 0]
        dx = bond_targets[b, i, 1]
        dy = bond_targets[b, i, 2]

        # 周期性边界条件
        nx = (x + dx) % Nx
        ny = (y + dy) % Ny

        S_j = spins[nx, ny, neighbor_b]
        J = bond_matrices[b, i]

        # S_val . (J . S_j) 手动展开极快
        J_Sj_x = J[0,0]*S_j[0] + J[0,1]*S_j[1] + J[0,2]*S_j[2]
        J_Sj_y = J[1,0]*S_j[0] + J[1,1]*S_j[1] + J[1,2]*S_j[2]
        J_Sj_z = J[2,0]*S_j[0] + J[2,1]*S_j[1] + J[2,2]*S_j[2]

        E += S_val[0]*J_Sj_x + S_val[1]*J_Sj_y + S_val[2]*J_Sj_z

    # 2. 磁晶各向异性能 (A_ani < 0 对应易磁化轴为 Z 轴)
    E += A_ani[b] * (S_val[2] * S_val[2])

    # 3. 塞曼能 (已在外部转为 meV)
    E -= (B_field_meV[0]*S_val[0] + B_field_meV[1]*S_val[1] + B_field_meV[2]*S_val[2])

    return E

@njit(cache=True)
def _mc_step_numba(spins, T_meV, Nx, Ny, Nb, num_bonds, bond_targets, bond_matrices,
                   A_ani, B_field_meV, proposal_angle, global_move_probability, rnd):
    """执行一个 Monte Carlo sweep。

    随机数全部由 Python 层预生成（per-object RNG，可复现），本内核
    不调用任何 RNG —— 避免了 numba 进程级共享 RNG 状态导致的
    对象间串扰（numpy 的 np.random.seed 无法隔离 numba 内核状态）。

    局部提案是单位球面上的随机转动：转角分布与方位角均为对称分布，
    因而可直接使用 Metropolis 判据。不要使用"加笛卡尔随机矢量再归一化"
    的提案；该提案通常不是对称的，须配 Metropolis--Hastings 修正。
    """
    N_total = Nx * Ny * Nb
    accepted = 0

    for i in range(N_total):
        # 随机挑选一个格点（rnd 各列 ∈ [0,1)）
        x = int(rnd[i, 0] * Nx)
        y = int(rnd[i, 1] * Ny)
        b = int(rnd[i, 2] * Nb)

        S_old = spins[x, y, b]

        # 少量全局均匀提案与局部球面旋转的混合仍然保持详细平衡。
        if rnd[i, 3] < global_move_probability:
            z = rnd[i, 4] * 2.0 - 1.0
            phi = rnd[i, 5] * 2.0 * np.pi
            sin_theta = np.sqrt(1.0 - z*z)
            S_new = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), z])
        else:
            # 在 S_old 的切平面取均匀方位角，再作对称的随机转动。
            if abs(S_old[2]) < 0.9:
                e1x = -S_old[1]
                e1y = S_old[0]
                e1z = 0.0
            else:
                e1x = 0.0
                e1y = -S_old[2]
                e1z = S_old[1]
            e1norm = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
            e1x /= e1norm
            e1y /= e1norm
            e1z /= e1norm
            # e2 = S_old x e1
            e2x = S_old[1]*e1z - S_old[2]*e1y
            e2y = S_old[2]*e1x - S_old[0]*e1z
            e2z = S_old[0]*e1y - S_old[1]*e1x

            alpha = (rnd[i, 4] * 2.0 - 1.0) * proposal_angle
            phi = rnd[i, 5] * 2.0 * np.pi
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            tx = np.cos(phi)*e1x + np.sin(phi)*e2x
            ty = np.cos(phi)*e1y + np.sin(phi)*e2y
            tz = np.cos(phi)*e1z + np.sin(phi)*e2z
            S_new = np.array([ca*S_old[0] + sa*tx,
                              ca*S_old[1] + sa*ty,
                              ca*S_old[2] + sa*tz])

        # 计算能量差 (meV)
        E_old = _local_energy_numba(x, y, b, S_old, spins, Nx, Ny, num_bonds, bond_targets, bond_matrices, A_ani, B_field_meV)
        E_new = _local_energy_numba(x, y, b, S_new, spins, Nx, Ny, num_bonds, bond_targets, bond_matrices, A_ani, B_field_meV)
        dE = E_new - E_old

        # Metropolis 判据
        if dE <= 0.0 or rnd[i, 6] < np.exp(-dE / max(T_meV, 1e-12)):
            spins[x, y, b, 0] = S_new[0]
            spins[x, y, b, 1] = S_new[1]
            spins[x, y, b, 2] = S_new[2]
            accepted += 1

    return accepted / N_total

@njit(cache=True)
def _total_energy_numba(spins, Nx, Ny, num_bonds, bond_targets, bond_matrices, A_ani, B_field_meV):
    """总能量（meV）。每条双向存储的键只计一次（0.5 因子）。"""
    Nb = spins.shape[2]
    E = 0.0
    for x in range(Nx):
        for y in range(Ny):
            for b in range(Nb):
                S_i = spins[x, y, b]
                n_bonds = num_bonds[b]
                for i in range(n_bonds):
                    neighbor_b = bond_targets[b, i, 0]
                    dx = bond_targets[b, i, 1]
                    dy = bond_targets[b, i, 2]
                    S_j = spins[(x + dx) % Nx, (y + dy) % Ny, neighbor_b]
                    J = bond_matrices[b, i]
                    J_Sj_x = J[0,0]*S_j[0] + J[0,1]*S_j[1] + J[0,2]*S_j[2]
                    J_Sj_y = J[1,0]*S_j[0] + J[1,1]*S_j[1] + J[1,2]*S_j[2]
                    J_Sj_z = J[2,0]*S_j[0] + J[2,1]*S_j[1] + J[2,2]*S_j[2]
                    E += 0.5 * (S_i[0]*J_Sj_x + S_i[1]*J_Sj_y + S_i[2]*J_Sj_z)
                E += A_ani[b] * S_i[2] * S_i[2]
                E -= (B_field_meV[0]*S_i[0] + B_field_meV[1]*S_i[1] + B_field_meV[2]*S_i[2])
    return E

# ==========================================
# 面向对象的物理框架区 (用户接口)
# ==========================================
class Lattice:
    def __init__(self, a_vecs, basis, Nx, Ny):
        self.a_vecs = np.array(a_vecs, dtype=np.float64)
        self.basis = np.array(basis, dtype=np.float64)
        self.Nb = len(basis)
        self.Nx = Nx
        self.Ny = Ny
        self.N_total = Nx * Ny * self.Nb

    def get_cartesian_coords(self):
        coords = np.zeros((self.Nx, self.Ny, self.Nb, 2))
        for x in range(self.Nx):
            for y in range(self.Ny):
                for b in range(self.Nb):
                    f1, f2 = self.basis[b]
                    r = (x + f1) * self.a_vecs[0] + (y + f2) * self.a_vecs[1]
                    coords[x, y, b] = r
        return coords

class Hamiltonian:
    def __init__(self, Nb, A_ani, B_field=np.array([0.0, 0.0, 0.0])):
        """
        A_ani: 磁晶各向异性 (meV)
        B_field: 外磁场矢量 (Tesla)
        """
        self.Nb = Nb
        self.A = np.ones(Nb, dtype=np.float64) * A_ani if np.isscalar(A_ani) else np.array(A_ani, dtype=np.float64)

        # 将传入的磁场(Tesla)转化为塞曼能量(meV)
        self.B_field_meV = np.array(B_field, dtype=np.float64) * MU_S_MEV_PER_T

        self.bonds = [[] for _ in range(Nb)]

    def add_bond(self, b1, b2, offset, J_matrix):
        """ J_matrix 单位要求为 meV """
        dx, dy = offset
        J = np.array(J_matrix, dtype=np.float64)
        self.bonds[b1].append((b2, dx, dy, J))
        # 自动添加反向作用（使用 J 矩阵的转置，完美符合海森堡模型物理逻辑）
        if b1 != b2 or dx != 0 or dy != 0:
            self.bonds[b2].append((b1, -dx, -dy, J.T))

    def build_numba_arrays(self):
        max_bonds = max([len(b) for b in self.bonds]) if self.Nb > 0 else 0
        max_bonds = max(1, max_bonds)

        self.num_bonds = np.zeros(self.Nb, dtype=np.int32)
        self.bond_targets = np.zeros((self.Nb, max_bonds, 3), dtype=np.int32)
        self.bond_matrices = np.zeros((self.Nb, max_bonds, 3, 3), dtype=np.float64)

        for b in range(self.Nb):
            self.num_bonds[b] = len(self.bonds[b])
            for i, (neighbor_b, dx, dy, J) in enumerate(self.bonds[b]):
                self.bond_targets[b, i, 0] = neighbor_b
                self.bond_targets[b, i, 1] = dx
                self.bond_targets[b, i, 2] = dy
                self.bond_matrices[b, i] = J

class MonteCarlo2D:
    def __init__(self, lattice, hamiltonian, seed=None):
        """seed 不为 None 时固定随机序列（对象级独立 RNG），保证可复现。

        内核不调用任何 RNG：每轮 sweep 的随机数由 self._rng 预生成后
        传入 numba 内核，因此不同对象（不同 seed）完全隔离，互不串扰。
        """
        self._rng = np.random.default_rng(seed)
        self.lat = lattice
        self.ham = hamiltonian
        self.ham.build_numba_arrays()
        self.spins = self._random_spins()
        # 接受率自适应状态：None = 使用用户给定角；否则自动调节
        self._auto_angle = True
        self._proposal_angle = None
        self._sweeps_since_adjust = 0

    def _random_spins(self):
        phi = self._rng.uniform(0, 2 * np.pi, size=(self.lat.Nx, self.lat.Ny, self.lat.Nb))
        costheta = self._rng.uniform(-1, 1, size=(self.lat.Nx, self.lat.Ny, self.lat.Nb))
        sintheta = np.sqrt(1 - costheta**2)
        # 使用 np.ascontiguousarray 确保内存连续，最大化 Numba 速度
        spins = np.stack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta), axis=-1).astype(np.float64)
        return np.ascontiguousarray(spins)

    def _adapt_angle(self, acceptance):
        """按接受率反馈调节提案角，目标区间 [0.30, 0.60]。"""
        self._sweeps_since_adjust += 1
        if self._sweeps_since_adjust < _ADAPT_EVERY:
            return
        self._sweeps_since_adjust = 0
        if self._proposal_angle is None:
            self._proposal_angle = 0.2
        if acceptance > _ACCEPT_TARGET_HI:
            self._proposal_angle = min(_ANGLE_MAX, self._proposal_angle * _ANGLE_ADJUST)
        elif acceptance < _ACCEPT_TARGET_LO:
            self._proposal_angle = max(_ANGLE_MIN, self._proposal_angle / _ANGLE_ADJUST)

    def mc_step(self, T_K, proposal_angle=None, global_move_probability=0.02):
        """执行一个 sweep；温度输入为 K，返回接受率。

        ``proposal_angle`` 为局部自旋转角上限（弧度）。默认 None 时启用
        接受率自适应（目标 ~30-60%），无需手动调节。
        """
        T_meV = float(T_K) * KB_MEV_PER_K
        if T_meV < 0:
            raise ValueError("温度不能为负")
        if proposal_angle is None:
            if self._proposal_angle is None:
                # 初始角：随温度缩小的保守值
                self._proposal_angle = min(0.30, max(0.02, 0.20 * np.sqrt(T_meV)))
            self._auto_angle = True
        else:
            self._auto_angle = False
            self._proposal_angle = float(proposal_angle)
        if not 0.0 <= global_move_probability <= 1.0:
            raise ValueError("global_move_probability 必须在 [0, 1] 内")
        # 预生成整轮随机数（7 列：x, y, b, 提案类型, 角度/高度, 方位角, 接受判定）
        N_total = self.lat.N_total
        rnd = self._rng.random((N_total, 7))
        acc = _mc_step_numba(
            self.spins, T_meV, self.lat.Nx, self.lat.Ny, self.lat.Nb,
            self.ham.num_bonds, self.ham.bond_targets, self.ham.bond_matrices,
            self.ham.A, self.ham.B_field_meV, float(self._proposal_angle),
            float(global_move_probability), rnd
        )
        if self._auto_angle:
            self._adapt_angle(acc)
        return acc

    def get_magnetization(self):
        M_vec = np.sum(self.spins, axis=(0,1,2)) / self.lat.N_total
        return np.linalg.norm(M_vec), M_vec

    def total_energy(self):
        """返回总哈密顿量（meV）。每条双向存储的键只计一次（numba 加速）。"""
        return _total_energy_numba(
            self.spins, self.lat.Nx, self.lat.Ny,
            self.ham.num_bonds, self.ham.bond_targets, self.ham.bond_matrices,
            self.ham.A, self.ham.B_field_meV,
        )

    def spin_structure_factor(self):
        """自旋结构因子 S(q) = |Σ_r S(r) e^{iq·r}|²/N（含子格相位）。

        返回 (q1, q2, S) 三个网格数组。q 是 FFT 频域索引（整数，
        实际波矢 = 2π·(q1/Nx)·b1* + 2π·(q2/Ny)·b2*，b* 为倒格矢）。
        用途：识别磁序——FM 在 q=(0,0) 有峰；螺旋/自旋密度波在
        q=±q*；skyrmion 晶格出现 q=0 强峰 + 晶格 Bragg 峰。
        """
        Nx, Ny, Nb, _ = self.spins.shape
        S_sum = np.zeros((Nx, Ny), dtype=np.complex128)
        for b in range(Nb):
            f1, f2 = self.lat.basis[b]
            # 子格位置相位：S(q) = Σ_b e^{iq·τ_b} Σ_r S_b(r) e^{iq·r}
            phase = np.exp(-2j*np.pi*(f1*np.arange(Nx)[:, None] + f2*np.arange(Ny)[None, :]))
            for comp in range(3):
                F = np.fft.fft2(self.spins[:, :, b, comp] * phase)
                S_sum += F * np.conj(F)
        S = S_sum.real / (Nx * Ny * Nb)
        q1 = np.fft.fftfreq(Nx) * Nx   # 整数索引 [-Nx/2, Nx/2)
        q2 = np.fft.fftfreq(Ny) * Ny
        return q1, q2, S

    def magnon_spectrum(self, k_path, S=1.0, B_field=(0.0, 0.0, 0.0)):
        """共线铁磁磁振子色散 ω(k)（线性自旋波理论，Holstein–Primakoff 一阶）。

        公式（已用 1D 4-环 S=1..3 精确对角化校准，见 references）：
          ω_k = S·|J|·Σ_δ(1 − cos k·δ) + 2S|A| + μ_s·B_z   （各向同性 J；多子格取
          bond 矩阵的 J_zz 与横向 (J_xx+J_yy)/2；A<0 易轴给能隙 2S|A|；B_z 给 Zeeman 隙）

        k_path : 分数坐标 k 列表，如 [(0,0), (1/3,1/3), (1/2,0), (0,0)]
        S      : 自旋量子数（LSWT 是 S→∞ 极限，S=1 时偏差 ~1/(2S)）
        B_field: 纵向外场 (Tesla)

        限制：仅共线 FM 基态；反对称 J（DMI）忽略——界面 DMI 使基态倾斜，
        共线 LSWT 不合法（给出伪线性项）；非共线/AFM 基态不支持。

        返回 (k_list, omega)：omega[nk, nband] 单位 meV。
        """
        Nb = self.lat.Nb
        Bz_meV = B_field[2] * MU_S_MEV_PER_T
        omegas = []
        for (kx, ky) in k_path:
            M = np.zeros((Nb, Nb), dtype=complex)
            for mu in range(Nb):
                f1m, f2m = self.lat.basis[mu]
                for (nu, dx, dy, Jm) in self.ham.bonds[mu]:
                    Jzz = Jm[2, 2]
                    Jxy = 0.5 * (Jm[0, 0] + Jm[1, 1])
                    f1n, f2n = self.lat.basis[nu]
                    # 相位：r_μ − r_ν = (τ_μ−τ_ν) − δ（分数坐标）
                    ph = 2*np.pi*(kx*(f1m-f1n-dx) + ky*(f2m-f2n-dy))
                    # 每 directed bond 的 ½（双向存储的双计数修正）
                    M[mu, mu] += 0.5 * (-S * Jzz)              # a†_μ a_μ
                    M[nu, nu] += 0.5 * (-S * Jzz)              # a†_ν a_ν
                    M[mu, nu] += 0.5 * S * (Jzz + Jxy) * np.exp(-1j*ph)
            # 单离子各向异性（A<0 易轴 → +2S|A|）与外场（Zeeman）
            for mu in range(Nb):
                M[mu, mu] += -2.0 * S * self.ham.A[mu] + Bz_meV
            w = np.linalg.eigvalsh(M)
            w = np.maximum(w, 0.0)          # 数值小负值归零（Goldstone）
            omegas.append(w)
        return list(k_path), np.array(omegas)

    def run_magnetocaloric(self, T_list, B_list, equip_steps, calc_steps,
                           sample_interval=1, output_file="mce_results.txt",
                           molar_mass_g_per_mol=None):
        """磁热效应：多场温度扫描，输出熵变 ΔS_M(T) 与绝热温变 ΔT_ad(T)。

        物理（经典连续自旋，k_B=1）：
          ΔS_M(T,ΔB) = ∫₀^{ΔB} (∂M/∂T)_{B'} dB'     （麦克斯韦关系，主算法）
          S(T,B) = ln(4π) − ∫₀^{β} (E−E₀) dβ' + β(E−E₀)  （绝对熵，附；低 T 平衡不足时不可靠）
          ΔT_ad(T₁) = T₂ − T₁，其中 S(T₂,B) = S(T₁,0)   （等熵构造，严格）
          C(T) = Var(E)/(k_B T²)                      （涨落公式）

        单位：
          molar_mass_g_per_mol 给定时（每磁性自旋的摩尔质量，如 CrI₃=432.7），
          ΔS_M 与 C 输出为 J/(kg·K)（换算因子 R/M_mol×10³，R=8.3145 J/mol·K）；
          None 时保持 k_B/自旋（ΔT_ad 恒为 K）。

        返回 (T_grid, B_list, S_matrix, dS_M, C_matrix, dT_ad)：
          S_matrix[nT, nB]   每自旋绝对熵（低 T 仅供参考）
          dS_M[nT, nB-1]     列 j = ΔS_M 相对 B=0（麦克斯韦关系）
          C_matrix[nT, nB]   每自旋热容
          dT_ad[nT, nB-1]    列 j = ΔT_ad 相对 B=0（等熵构造 S(T₂,B)=S(T₁,0)；无解为 NaN）
          （molar_mass 给定时 S/dS_M/C 单位 J/(kg·K)，否则 k_B/自旋）
        """
        import numpy as np
        T_list = np.asarray(T_list, float)
        B_list = np.asarray(B_list, float)
        nT, nB = len(T_list), len(B_list)

        U = np.zeros((nT, nB))      # 每自旋平均能量 (meV)
        U2 = np.zeros((nT, nB))     # 每自旋能量平方平均
        M = np.zeros((nT, nB))      # |M|

        print(f"--- MCE: {nT} 温度 × {nB} 场 ---")
        for j, Bz in enumerate(B_list):
            self.ham.B_field_meV = np.array([0.0, 0.0, Bz], dtype=np.float64) * MU_S_MEV_PER_T
            for i, T in enumerate(T_list):
                self._validate_sampling(float(T), equip_steps, calc_steps, sample_interval)
                for _ in range(equip_steps):
                    self.mc_step(T)
                N = self.lat.N_total
                e_acc = e2_acc = 0.0
                for _ in range(calc_steps):
                    for _ in range(sample_interval):
                        self.mc_step(T)
                    E = self.total_energy() / N
                    e_acc += E
                    e2_acc += E * E
                U[i, j] = e_acc / calc_steps
                U2[i, j] = e2_acc / calc_steps
                M[i, j] = self.get_magnetization()[0]
                print(f"  B={Bz:5.2f} T  T={T:6.1f} K  <E>={U[i,j]:7.4f} meV/spin  M={M[i,j]:.3f}")

        # ---- 熵积分：S(T,B) = ln(4π) − ∫₀^β (E−E₀) dβ' + β(E−E₀)（每自旋, k_B=1）----
        # 注意：不能直接用 ∫E dβ' 的形式——低 T 时 ∫ 与 βE 是发散量的差，
        # 必须用基态能量 E₀(B) 作参考（E−E₀ 在 β→∞ 收敛）。
        beta = 1.0 / (KB_MEV_PER_K * T_list)          # 1/meV
        order = np.argsort(beta)                        # β 升序（= T 降序）
        beta_s, T_s, U_s = beta[order], T_list[order], U[order]
        S = np.zeros_like(U)
        E_inf = np.mean(self.ham.A) / 3.0   # 高温极限能量 E(∞) = ⟨A⟩/3（每自旋）
        for j in range(nB):
            # 基态能量外推：2D 经典自旋波 E−E₀ ∝ T²，用 E = E₀ + c·T² 拟合
            # 最低 4 个 T（含 T 线性项的拟合会被低 T 噪声拉偏 → S 出现负值）
            n_ex = min(4, nT)
            Tf = T_s[-n_ex:]
            Uf = U_s[-n_ex:, j]
            c, E0 = np.polyfit(Tf**2, Uf, 1)
            # 物理约束：基态能量 E₀ ≤ U(T_min)（T>0 平均能量不低于基态）
            E0 = min(E0, Uf[0])
            # (E−E₀) 在 β 网格上梯形积分；β=0 端点补 E(∞)−E₀（漏掉 [0,β₁] 段
            # 会带来 ~1 k_B/自旋的系统偏移——解析验证发现）
            dU = U_s[:, j] - E0
            beta_grid = np.concatenate([[0.0], beta_s])
            dU_grid = np.concatenate([[E_inf - E0], dU])
            integ = np.concatenate([[0.0], np.cumsum(
                0.5 * (dU_grid[1:] + dU_grid[:-1]) * np.diff(beta_grid))])
            S[:, j] = np.log(4.0 * np.pi) - integ[1:] + beta_s * dU
        # 还原 T 顺序
        inv = np.argsort(order)
        S = S[inv]

        # ---- 热容（涨落公式，每自旋, k_B 单位）----
        # C_per_spin = N·Var(e)/(k_B T²)，e = 每自旋能量（Var 需 ×N² 还原总能量方差再 /N）
        kB_T2 = KB_MEV_PER_K * T_list**2
        C = self.lat.N_total * (U2 - U**2) / kB_T2[:, None]

        # ---- ΔS_M：麦克斯韦关系 ΔS_M(T) = ∫ (∂M/∂T)_B dB ----
        # 比绝对熵积分稳：无 E₀ 外推、无 β 尾巴；T→0 时 ∂M/∂T→0 自动归零。
        # T 升序排列后中心差分，再做 3 点滑动平均压噪声。
        T_asc = np.argsort(T_list)
        M_asc = M[T_asc]                       # (nT, nB)，T 升序
        T_a = T_list[T_asc]
        dM_dT = np.zeros_like(M_asc)
        dM_dT[1:-1] = (M_asc[2:] - M_asc[:-2]) / (T_a[2:] - T_a[:-2])[:, None]
        dM_dT[0] = (M_asc[1] - M_asc[0]) / (T_a[1] - T_a[0])
        dM_dT[-1] = (M_asc[-1] - M_asc[-2]) / (T_a[-1] - T_a[-2])
        # 5 点加权滑动平均 [1,4,6,4,1]/16（边界反射）；3 点对高 T 噪声压不住
        dM_dT_s = np.zeros_like(dM_dT)
        for i in range(nT):
            idx = np.clip([i-2, i-1, i, i+1, i+2], 0, nT-1)
            dM_dT_s[i] = np.average(dM_dT[idx], axis=0, weights=[1, 4, 6, 4, 1])
        # 对 B 梯形积分：ΔS_M(T, B_j) = ∫₀^{B_j} (∂M/∂T) dB'
        dS_M = np.zeros((nT, nB - 1))           # 列 j = B_{j+1} 相对 B=0 的熵变
        for j in range(nB - 1):
            dS_M[:, j] = np.trapezoid(
                dM_dT_s[:, :j + 2], B_list[:j + 2], axis=1)
        inv = np.argsort(T_asc)
        dS_M = dS_M[inv]

        # ---- ΔT_ad：等熵构造（严格）----
        # 绝热加场 B：S(T₂, B) = S(T₁, 0) → ΔT_ad = T₂ − T₁
        # 从绝对熵曲线反插值（比近似 −T·ΔS_M/C 精确，不受 C 噪声影响）。
        T_asc2 = np.argsort(T_list)
        Ta2 = T_list[T_asc2]
        Sa2 = S[T_asc2]                          # S 随 T 单调增
        dT_ad = np.full((nT, nB - 1), np.nan)
        for j in range(nB - 1):
            for i in range(nT):
                s0 = Sa2[i, 0]
                if Sa2[0, j + 1] <= s0 <= Sa2[-1, j + 1]:
                    T2 = np.interp(s0, Sa2[:, j + 1], Ta2)
                    dT_ad[i, j] = T2 - Ta2[i]
        inv2 = np.argsort(T_asc2)
        dT_ad = dT_ad[inv2]

        # ---- 单位换算：k_B/自旋 → J/(kg·K)（molar_mass_g_per_mol 给定时）----
        # 换算因子 = R/M_mol×10³（R = k_B·N_A = 8.3145 J/(mol·K)）
        if molar_mass_g_per_mol is not None:
            conv = 8.314462618 / (molar_mass_g_per_mol * 1e-3)   # J/(kg·K) per k_B/spin
            S = S * conv
            dS_M = dS_M * conv
            C = C * conv

        if output_file is not None:
            unit = "J_per_kg_per_K" if molar_mass_g_per_mol is not None else "kB_per_spin"
            with open(output_file, "w") as f:
                f.write(f"# T(K) B(T) E_meV_per_spin M C_{unit} S_abs_{unit} "
                        f"dS_M_relB0_{unit} dT_ad_relB0_K\n")
                f.write("# dS_M/dT_ad 列 = 相对 B=0 的磁熵变/绝热温变；"
                        "dS_M 用麦克斯韦关系，dT_ad 用等熵构造\n")
                for i, T in enumerate(T_list):
                    for j in range(nB):
                        f.write(f"{T:.4f} {B_list[j]:.4f} {U[i,j]:.8f} {M[i,j]:.6f} "
                                f"{C[i,j]:.8f} {S[i,j]:.8f} "
                                f"{(dS_M[i,j-1] if j>0 else 0.0):.8f} "
                                f"{(dT_ad[i,j-1] if j>0 else 0.0):.8f}\n")
            print(f"结果已保存至 '{output_file}'")

        return T_list, B_list, S, dS_M, C, dT_ad

    def magnetic_structure_analysis(self, q_threshold=0.30, m_fm_threshold=0.85):
        """自动识别当前自旋构型的磁结构，并提取磁性单胞。

        流程：
          1. 平均磁化 |m| —— |m| > m_fm_threshold → FM（单胞 1×1）
          2. S(q) 峰检测（排除 Γ）——无峰 → PM
          3. 分类：单 q* → AFM（半格点 q）/ helical（任意 q）；
             多 q* 且拓扑荷 Q ≠ 0 → skyrmion_lattice；多 q* 且 Q ≈ 0 → multi-q
          4. 磁单胞提取：求所有满足 q*·R ∈ ℤ 的最小线性无关格点
             R = (m,n)（即磁平移对称性），构造单胞基矢矩阵并抽取单胞自旋。

        返回 dict：
          order          : 'FM' | 'AFM' | 'helical' | 'skyrmion_lattice' | 'multi-q' | 'PM'
          magnetization  : |m|（归一化）
          q_stars        : 分数坐标 q* 列表（不含 ± 重复，不含 Γ）
          top_charge     : 拓扑荷 Q（skyrmion 检测）
          cell_matrix    : 磁单胞基矢 [[m1,n1],[m2,n2]]（超胞格点坐标）
          cell_spins     : (N1, N2, Nb, 3) 单胞内自旋构型
          cell_repeats   : 超胞 = cell_repeats[0]×cell_repeats[1] 个磁单胞
        """
        Nx, Ny, Nb, _ = self.spins.shape
        m_norm, _ = self.get_magnetization()
        q1, q2, S = self.spin_structure_factor()
        Q = 0.0
        # 长程有序判据：峰强度须显著高于背景（随机/热噪声无此特征）
        S_med = np.median(S)
        ordered = S.max() > 5.0 * S_med

        # --- 1. 峰检测（3×3 邻域极大，排除 Γ；需满足长程有序判据）---
        Smax = S.max()
        peaks = []
        if ordered:
            for i in range(Nx):
                for j in range(Ny):
                    if i == 0 and j == 0:
                        continue
                    s = S[i, j]
                    if s < q_threshold * Smax:
                        continue
                    # 3×3 邻域极大（PBC）
                    if all(s >= S[(i+di) % Nx, (j+dj) % Ny]
                           for di in (-1, 0, 1) for dj in (-1, 0, 1)
                           if not (di == 0 and dj == 0)):
                        peaks.append((i, j))
        # 去除 ±q 重复（保留一半）
        def canon(i, j):
            # 规范化到 [0, Nx)×[0, Ny) 的 q 对：取 (i,j) 与 (-i,-j) 的字典序较小者
            return min((i % Nx, j % Ny), ((-i) % Nx, (-j) % Ny))
        q_stars = []
        seen = set()
        for i, j in peaks:
            c = canon(i, j)
            if c not in seen:
                seen.add(c)
                q_stars.append(c)
        q_stars.sort()

        # --- 2. 分类 ---
        if m_norm > m_fm_threshold:
            order = "FM"
        elif not ordered:
            order = "PM"
        else:
            try:
                Q = self.topological_charge()
            except NotImplementedError:
                pass
            half_grid = lambda i, j: (2*i % Nx == 0) and (2*j % Ny == 0)
            if len(q_stars) == 1:
                i, j = q_stars[0]
                order = "AFM" if half_grid(i, j) else "helical"
            else:
                order = "skyrmion_lattice" if abs(Q) > 0.3 else "multi-q"

        # --- 3. 磁单胞提取（q*·R ∈ ℤ 的所有 R 的子格）---
        cell_matrix, cell_spins, nrep, cell_sites = self._extract_magnetic_cell(q_stars, order)

        return {
            "order": order,
            "magnetization": m_norm,
            "q_stars": q_stars,
            "top_charge": Q if order in ("skyrmion_lattice", "multi-q") else 0.0,
            "cell_matrix": cell_matrix,
            "cell_spins": cell_spins,
            "cell_sites": cell_sites,
            "cell_repeats": nrep,
        }

    def _extract_magnetic_cell(self, q_stars, order):
        """从 q* 构造磁单胞：解 i·m·Ny + j·n·Nx ≡ 0 (mod Nx·Ny)（对所有峰）。"""
        Nx, Ny, Nb, _ = self.spins.shape
        if order == "FM" or not q_stars:
            # 单胞 = 1×1（FM）或整超胞（PM 无周期，返回整胞）
            if order == "FM":
                return [[1, 0], [0, 1]], self.spins[:1, :1].copy(), Nx * Ny, [(0, 0)]
            return [[Nx, 0], [0, Ny]], self.spins.copy(), 1, [(x, y) for x in range(Nx) for y in range(Ny)]
        L = Nx * Ny
        mods = []
        for (i, j) in q_stars:
            mods.append(((i * Ny) % L, (j * Nx) % L))
        # 收集所有满足同余条件的 R=(m,n)，按范数排序 → 最短线性无关对
        sols = []
        rng_m = range(1, max(Nx, Ny) + 1)
        rng_n = range(-max(Nx, Ny), max(Nx, Ny) + 1)
        for m in rng_m:
            for n in rng_n:
                if all((a * m + b * n) % L == 0 for (a, b) in mods):
                    sols.append((m, n))
        if len(sols) < 2:
            # 兜底：对角超胞（每峰分母 lcm）
            from math import gcd
            d1 = d2 = 1
            for (i, j) in q_stars:
                if i:
                    d1 = d1 * (Nx // gcd(i, Nx)) // gcd(d1, Nx // gcd(i, Nx))
                if j:
                    d2 = d2 * (Ny // gcd(j, Ny)) // gcd(d2, Ny // gcd(j, Ny))
            sols = [(d1, 0), (0, d2)]
        sols.sort(key=lambda v: abs(v[0]) + abs(v[1]))
        basis = [sols[0]]
        for v in sols[1:]:
            if all(v[0]*w[1] - v[1]*w[0] != 0 for w in basis):
                basis.append(v)
            if len(basis) >= 2:
                break
        # LLL 约化（2D 简化版：使基矢变短、近正交）
        v1 = np.array(basis[0], dtype=int)
        v2 = np.array(basis[1], dtype=int)
        for _ in range(16):
            # 用 v2 修正 v1
            mu = round(np.dot(v1, v2) / np.dot(v2, v2))
            if mu != 0:
                v1 = v1 - mu * v2
            # 交换使 |v1| ≤ |v2|
            if np.dot(v1, v1) > np.dot(v2, v2):
                v1, v2 = v2, v1
            if all(v == 0 for v in (v1[0], v1[1])):
                v1 = np.array([1, 0])
            if all(v == 0 for v in (v2[0], v2[1])):
                v2 = np.array([0, 1])
        (m1, n1), (m2, n2) = tuple(v1), tuple(v2)
        # 单胞格点数与超胞内单胞重复数
        det = abs(m1 * n2 - n1 * m2)
        nrep = (Nx * Ny) // max(1, det) if det else 1
        # 单胞格点 = 超胞格点按磁平移等价类（ΔR 满足所有 q*·ΔR ∈ ℤ）分组的代表元
        classes = {}   # rep -> [members]
        reps = []
        for x in range(Nx):
            for y in range(Ny):
                rep = None
                for r0 in reps:
                    dx, dy = x - r0[0], y - r0[1]
                    if all((a*dx + b*dy) % L == 0 for (a, b) in mods):
                        rep = r0
                        break
                if rep is None:
                    reps.append((x, y))
                    classes[(x, y)] = [(x, y)]
                else:
                    classes[rep].append((x, y))
        uniq = sorted(classes.keys())   # 每类取最小字典序代表
        if not uniq:
            uniq = [(0, 0)]
        cell = np.array([self.spins[x, y] for (x, y) in uniq])  # (ncell, Nb, 3)
        # 重排为网格形状（近似方阵）
        ncell = len(uniq)
        nx = int(round(np.sqrt(ncell)))
        ny = (ncell + nx - 1) // nx
        grid = np.zeros((nx, ny, Nb, 3))
        for k in range(ncell):
            grid[k // ny, k % ny] = cell[k]
        return [[m1, n1], [m2, n2]], grid, nrep, uniq

    def export_magnetic_cell_poscar(self, path="magnetic_cell.POSCAR", spin_scale=1.0,
                                    species=None, vacuum_z=1.0, write_magmom=True):
        """导出磁单胞为 VASP POSCAR（2D 晶格 + z 真空层）+ MAGMOM 磁构型。

        - 结构：磁单胞晶格矢量 = cell_matrix × 原胞 a_vecs（z 固定为真空层厚度）；
          原子坐标 = 单胞格点（超胞整数坐标 → 笛卡尔 → 单胞分数坐标，wrap 到 [0,1)）
        - 磁性：MAGMOM 每原子 3 分量 = 自旋单位矢量 × spin_scale（默认 1.0，即单位矢量；
          传 S 或 μ_B 值可缩放）
        - 多子格：每格点 Nb 个原子（basis 位置）
        - write_magmom=True 时同时写 <path>.magmom（可直接粘贴进 INCAR）

        返回 (poscar 文本, magmom 字符串)。
        """
        r = self.magnetic_structure_analysis()
        sites = r["cell_sites"]
        cs = r["cell_spins"]                     # (nx, ny, Nb, 3)
        Nb = cs.shape[2]
        coords_all = self.lat.get_cartesian_coords()

        (m1, n1), (m2, n2) = r["cell_matrix"]
        a1 = np.array([self.lat.a_vecs[0][0], self.lat.a_vecs[0][1], 0.0])
        a2 = np.array([self.lat.a_vecs[1][0], self.lat.a_vecs[1][1], 0.0])
        L1 = m1 * a1 + n1 * a2
        L2 = m2 * a1 + n2 * a2
        L3 = np.array([0.0, 0.0, vacuum_z])
        lat_mat = np.array([[L1[0], L2[0]], [L1[1], L2[1]]])   # 2×2（列 = 基矢）
        inv = np.linalg.inv(lat_mat)

        atoms = []      # (frac_x, frac_y, frac_z, spin)
        for k, (x, y) in enumerate(sites):
            for b in range(Nb):
                cart2 = coords_all[x, y, b]         # (2,)
                frac = inv @ cart2
                frac = frac % 1.0
                frac = np.where(frac > 1.0 - 1e-10, 0.0, frac)   # 浮点舍入保护
                spin = cs[k // cs.shape[1], k % cs.shape[1], b] * spin_scale
                atoms.append((frac[0], frac[1], 0.0, spin))
        n_atoms = len(atoms)

        sp = species if species is not None else ["Spin"] * n_atoms
        if len(sp) != n_atoms:
            sp = ["Spin"] * n_atoms

        lines = [f"Magnetic cell from Monte Carlo (order={r['order']}, Q={r['top_charge']:.3f}, "
                 f"cell={r['cell_matrix']})"]
        lines.append("1.0")
        for L in (L1, L2, L3):
            lines.append(f"  {L[0]: .6f} {L[1]: .6f} {L[2]: .6f}")
        lines.append(" ".join(sorted(set(sp))))
        lines.append(str(n_atoms))
        lines.append("Direct")
        for (fx, fy, fz, _s) in atoms:
            lines.append(f"  {fx: .8f} {fy: .8f} {fz: .8f}")

        magmom = " ".join(f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}" for (_f, _g, _h, s) in atoms)

        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w") as f:
                f.write(text)
            if write_magmom:
                with open(path + ".magmom", "w") as f:
                    f.write(f"MAGMOM = {magmom}\n")
        return text, magmom

    def topological_charge(self):
        """三角格/六角格上的离散 skyrmion 数（周期性边界）。

        使用 NN 三角形剖分：每个菱形原胞劈成两个逆时针 NN 三角形
        (s00,s10,s01) 与 (s10,s11,s01)，Berg--Lüscher 立体角公式。
        """
        if self.lat.Nb != 1:
            raise NotImplementedError("拓扑数诊断目前只适用于 Nb=1 的三角晶格")

        def solid_angle(a, b, c):
            numerator = np.dot(a, np.cross(b, c))
            denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
            return 2.0 * np.arctan2(numerator, denominator)

        q = 0.0
        for x in range(self.lat.Nx):
            xp = (x + 1) % self.lat.Nx
            for y in range(self.lat.Ny):
                yp = (y + 1) % self.lat.Ny
                s00 = self.spins[x, y, 0]
                s10 = self.spins[xp, y, 0]
                s01 = self.spins[x, yp, 0]
                s11 = self.spins[xp, yp, 0]
                q += solid_angle(s00, s10, s01)
                q += solid_angle(s10, s11, s01)
        return q / (4.0 * np.pi)

    # ---------------- 3. 功能模块 ----------------
    def run_skyrmion_annealing(self, T_init, T_final, steps_per_T, B_field):
        """
        T_init, T_final: Kelvin
        B_field: Tesla
        """
        print(f"--- Numba: Skyrmion 退火模拟 (B={B_field} Tesla) ---")
        B = np.atleast_1d(np.asarray(B_field, dtype=np.float64))
        if B.size == 1:
            B = np.array([0.0, 0.0, B[0]])     # 标量 → 沿 z
        elif B.size == 3:
            B = B
        else:
            raise ValueError("B_field 应为标量（沿 z）或 3 分量矢量")
        self.ham.B_field_meV = B * MU_S_MEV_PER_T

        T = T_init
        cooling_rate = 0.9

        print("首次调用 Numba JIT 编译中...")
        self.mc_step(T)
        print("编译完成！开始退火。")

        while T > T_final:
            acceptance = 0.0
            for _ in range(steps_per_T):
                acceptance += self.mc_step(T)
            M, _ = self.get_magnetization()
            print(f"T = {T:.4f} K | |M| = {M:.4f} | accept = {acceptance/steps_per_T:.3f}")
            T *= cooling_rate

        # 极低温下的最后弛豫 (采用 0.01 K 模拟接近绝对零度)
        print("执行极低温基态弛豫...")
        for _ in range(steps_per_T * 2):
            self.mc_step(0.01)

        print(f"最终总能量 = {self.total_energy():.6f} meV")
        if self.lat.Nb == 1:
            print(f"离散拓扑数 Q = {self.topological_charge():.6f}")

        coords = self.lat.get_cartesian_coords()
        data = []
        for x in range(self.lat.Nx):
            for y in range(self.lat.Ny):
                for b in range(self.lat.Nb):
                    rx, ry = coords[x, y, b]
                    sx, sy, sz = self.spins[x, y, b]
                    data.append([rx, ry, sx, sy, sz])
        np.savetxt("skyrmion_spins.txt", data, header="X Y Sx Sy Sz", fmt="%.6f")
        print("退火完成，结果已保存至 'skyrmion_spins.txt'。")

    @staticmethod
    def _validate_sampling(T, equip_steps, calc_steps, sample_interval):
        if T <= 0.0:
            raise ValueError("统计温度必须大于零")
        if equip_steps < 0 or calc_steps <= 0 or sample_interval <= 0:
            raise ValueError("equip_steps >= 0、calc_steps > 0、sample_interval > 0")

    def run_curie_temperature(self, T_list, equip_steps, calc_steps,
                              B_field=(0.0, 0.0, 0.0), sample_interval=1,
                              output_file="curie_results.txt"):
        """在固定外场下扫描温度，输出平均磁矩与磁化率。

        Parameters
        ----------
        T_list : array-like
            温度列表，单位 K。
        equip_steps : int
            每个温度下的平衡 sweep 数。
        calc_steps : int
            每个温度下记录磁化的次数。
        B_field : length-3 array-like
            固定外场 (Tesla)。默认 [0, 0, 0]，可用来研究有限场磁相变。
        sample_interval : int
            两次记录之间的 sweep 数；增大它可降低样本自相关。

        ``Chi_*_per_T`` 是单位自旋磁化 M=<S> 对 Tesla 的响应 dM/dB；
        ``Chi_*_per_meV`` 是对塞曼能变量 h=mu_s B 的响应 dM/dh。
        """
        B_field = np.asarray(B_field, dtype=np.float64)
        if B_field.shape != (3,):
            raise ValueError("B_field 必须是长度为 3 的 Tesla 矢量")

        # 显式设置外场，避免继承此前退火或磁滞计算留下的状态。
        self.ham.B_field_meV = B_field * MU_S_MEV_PER_T
        results = []
        print(f"--- Numba: 温度扫描 (B={B_field} Tesla) ---")

        for T in T_list:
            self._validate_sampling(float(T), equip_steps, calc_steps, sample_interval)
            for _ in range(equip_steps):
                self.mc_step(T)

            M_vec_samples = np.empty((calc_steps, 3), dtype=np.float64)
            E_samples = np.empty(calc_steps, dtype=np.float64)
            N = self.lat.N_total
            for i in range(calc_steps):
                for _ in range(sample_interval):
                    self.mc_step(T)
                _, M_vec = self.get_magnetization()
                M_vec_samples[i] = M_vec
                E_samples[i] = self.total_energy() / N

            M_mean_vec = np.mean(M_vec_samples, axis=0)
            M_abs_mean = np.mean(np.linalg.norm(M_vec_samples, axis=1))
            T_meV = float(T) * KB_MEV_PER_K
            chi_per_meV = self.lat.N_total / T_meV * (
                np.mean(M_vec_samples**2, axis=0) - M_mean_vec**2)
            # h = mu_s B，因此 d<M>/dB = mu_s d<M>/dh。
            chi_per_T = MU_S_MEV_PER_T * chi_per_meV
            # 热容（涨落公式，每自旋, k_B 单位）：C_per_spin = N·Var(e)/(k_B T²)
            C_kB = N * (np.mean(E_samples**2) - np.mean(E_samples)**2) / (KB_MEV_PER_K * T**2)

            results.append([T, M_abs_mean, *chi_per_T, *chi_per_meV, C_kB])
            print(f"T={T:.3f} K | <|M|>={M_abs_mean:.5f} | "
                  f"Chi_B=(x:{chi_per_T[0]:.5f}, y:{chi_per_T[1]:.5f}, "
                  f"z:{chi_per_T[2]:.5f}) 1/T | C={C_kB:.4f} kB/spin")

        if output_file is not None:
            np.savetxt(output_file, results,
                       header=("T(K) M_abs_mean_spin Chi_x_per_T Chi_y_per_T Chi_z_per_T "
                               "Chi_x_per_meV Chi_y_per_meV Chi_z_per_meV C_kB_per_spin"),
                       fmt="%.8f")
        return np.asarray(results)

    def run_hysteresis_loop(self, B_list, T, equip_steps, calc_steps,
                            sample_interval=1, output_file="hysteresis_loop.txt"):
        """沿 ``B_list`` 的顺序扫描磁场并输出时间平均 <M_z>。

        每个场点先平衡 ``equip_steps`` 个 sweep，再对 ``calc_steps`` 个
        记录点求平均。保留前一场点的末态，因而 ``B_list`` 的顺序定义了
        磁滞路径。外场固定沿 z 方向，单位 Tesla。
        """
        self._validate_sampling(float(T), equip_steps, calc_steps, sample_interval)
        results = []
        print(f"--- Numba: 磁滞回线 (T={T} K)，输出 <M_z> ---")

        for Bz in B_list:
            self.ham.B_field_meV = np.array([0.0, 0.0, Bz], dtype=np.float64) * MU_S_MEV_PER_T
            for _ in range(equip_steps):
                self.mc_step(T)

            Mz_samples = np.empty(calc_steps, dtype=np.float64)
            for i in range(calc_steps):
                for _ in range(sample_interval):
                    self.mc_step(T)
                Mz_samples[i] = self.get_magnetization()[1][2]

            Mz_mean = np.mean(Mz_samples)
            results.append([Bz, Mz_mean])
            print(f"B_z={Bz:.4f} T | <M_z>={Mz_mean:.6f}")

        np.savetxt(output_file, results, header="B_z(T) M_z_mean_spin", fmt="%.8f")
        return np.asarray(results)


# ==========================================
# 多 seed 并行工具（multiprocessing）
# ==========================================
def _curie_worker(args):
    """单个 seed 的完整温度扫描（进程内独立 RNG）。"""
    (seed, a_vecs, basis, Nx, Ny, ham_spec, T_list,
     equip_steps, calc_steps, sample_interval, B_field) = args

    lat = Lattice(a_vecs, basis, Nx, Ny)
    ham = Hamiltonian(Nb=len(basis), A_ani=ham_spec["A_ani"])
    for (b1, b2, off, J) in ham_spec["bonds"]:
        ham.add_bond(b1, b2, off, J)
    mc = MonteCarlo2D(lat, ham, seed=seed)
    res = mc.run_curie_temperature(T_list, equip_steps, calc_steps,
                                   B_field=B_field, sample_interval=sample_interval,
                                   output_file=None)
    return res


def run_curie_temperature_seeds(ham_spec, a_vecs, basis, Nx, Ny, T_list,
                                equip_steps, calc_steps, sample_interval=1,
                                B_field=(0.0, 0.0, 0.0), n_seeds=4, n_workers=None,
                                base_seed=0, output_file="curie_seeds_results.txt"):
    """多 seed 并行温度扫描：返回每温度的 mean±std（M 与 χ），可估计统计误差。

    ``ham_spec`` 形如 {"A_ani": ..., "bonds": [(b1, b2, offset, J_matrix), ...]}。
    """
    import multiprocessing as mp

    if n_workers is None:
        n_workers = min(n_seeds, mp.cpu_count())
    n_workers = min(n_workers, n_seeds)

    tasks = [(base_seed + s, a_vecs, basis, Nx, Ny, ham_spec, T_list,
              equip_steps, calc_steps, sample_interval, B_field)
             for s in range(n_seeds)]

    print(f"--- 多 seed 并行温度扫描: {n_seeds} seeds / {n_workers} workers ---")
    with mp.Pool(n_workers) as pool:
        all_res = pool.map(_curie_worker, tasks)

    all_res = np.asarray(all_res)          # (n_seeds, nT, 8)
    mean = all_res.mean(axis=0)
    std = all_res.std(axis=0, ddof=1)
    out = np.column_stack([mean[:, :2], std[:, :2], mean[:, 2:], std[:, 2:]])
    if output_file is not None:
        np.savetxt(output_file, out,
                   header=("T(K) M_mean M_std Chi_perT_mean(3) Chi_perT_std(3) "
                           "Chi_permeV_mean(3) Chi_permeV_std(3)"),
                   fmt="%.8f")
    print("完成。mean±std 已保存。")
    return mean, std, all_res
