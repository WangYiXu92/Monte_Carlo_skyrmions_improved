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
                    # on-site 键（b1==b2 且 δ=0）无反条目：不计 0.5（否则能量减半）
                    if neighbor_b == b and dx == 0 and dy == 0:
                        E += S_i[0]*J_Sj_x + S_i[1]*J_Sj_y + S_i[2]*J_Sj_z
                    else:
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
        # np.ndim 同时兼容 Python 标量与 np.float64（np.isscalar(np.float64) 返回 False）
        self.A = np.ones(Nb, dtype=np.float64) * A_ani if np.ndim(A_ani) == 0 else np.array(A_ani, dtype=np.float64)

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
        """自旋结构因子 S(q) = |Σ_b e^{iq·τ_b} Σ_r S_b(r) e^{iq·r}|²/N。

        子格相位乘在频域（e^{−2πi(q1·f1/Nx + q2·f2/Ny)}），子格间相干求和
        （先加后取模——分别取模丢失干涉项）。返回 (q1, q2, S) 整数频域索引
        （实际波矢 = 2π·(q1/Nx)·b1* + 2π·(q2/Ny)·b2*，b* 为倒格矢）。
        用途：识别磁序——FM 在 q=(0,0) 有峰；螺旋/自旋密度波在 q=±q*；
        skyrmion 晶格出现 q=0 强峰 + 晶格 Bragg 峰。
        """
        Nx, Ny, Nb, _ = self.spins.shape
        iq = np.arange(Nx)[:, None]
        jq = np.arange(Ny)[None, :]
        F_total = np.zeros((Nx, Ny, 3), dtype=np.complex128)
        for b in range(Nb):
            f1, f2 = self.lat.basis[b]
            phase = np.exp(-2j*np.pi*(f1*iq/Nx + f2*jq/Ny))
            for comp in range(3):
                F_total[:, :, comp] += np.fft.fft2(self.spins[:, :, b, comp]) * phase
        S = np.sum(F_total * np.conj(F_total), axis=2).real / (Nx * Ny * Nb)
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
        if len(B_list) == 0 or B_list[0] != 0.0:
            raise ValueError("B_list 必须以 0 T 开头（ΔS_M/ΔT_ad 相对 B=0）")
        nT, nB = len(T_list), len(B_list)

        U = np.zeros((nT, nB))      # 每自旋平均能量 (meV)
        U2 = np.zeros((nT, nB))     # 每自旋能量平方平均
        M = np.zeros((nT, nB))      # |M|

        print(f"--- MCE: {nT} 温度 × {nB} 场 ---")
        orig_B = self.ham.B_field_meV.copy()
        try:
            for j, Bz in enumerate(B_list):
                self.ham.B_field_meV = np.array([0.0, 0.0, Bz], dtype=np.float64) * MU_S_MEV_PER_T
                for i, T in enumerate(T_list):
                    self._validate_sampling(float(T), equip_steps, calc_steps, sample_interval)
                    for _ in range(equip_steps):
                        self.mc_step(T)
                    N = self.lat.N_total
                    e_acc = e2_acc = m_acc = 0.0
                    for _ in range(calc_steps):
                        for _ in range(sample_interval):
                            self.mc_step(T)
                        E = self.total_energy() / N
                        e_acc += E
                        e2_acc += E * E
                        # 麦克斯韦关系用 M_z（场沿 z）：|⟨S⟩| 在高 T 含 1/√N 各向同性偏置
                        m_acc += self.get_magnetization()[1][2]
                    U[i, j] = e_acc / calc_steps
                    U2[i, j] = e2_acc / calc_steps
                    M[i, j] = m_acc / calc_steps
                    print(f"  B={Bz:5.2f} T  T={T:6.1f} K  <E>={U[i,j]:7.4f} meV/spin  M={M[i,j]:.3f}")
        finally:
            self.ham.B_field_meV = orig_B

        # ---- 熵积分：S(T,B) = ln(4π) − ∫₀^β (E−E₀) dβ' + β(E−E₀)（每自旋, k_B=1）----
        # 注意：不能直接用 ∫E dβ' 的形式——低 T 时 ∫ 与 βE 是发散量的差，
        # 必须用基态能量 E₀(B) 作参考（E−E₀ 在 β→∞ 收敛）。
        beta = 1.0 / (KB_MEV_PER_K * T_list)          # 1/meV
        order = np.argsort(beta)                        # β 升序（= T 降序）
        beta_s, T_s, U_s = beta[order], T_list[order], U[order]
        S = np.zeros_like(U)
        E_inf = np.mean(self.ham.A) / 3.0   # 高温极限能量 E(∞) = ⟨A⟩/3（每自旋）
        for j in range(nB):
            # 基态能量外推：经典自旋波 equipartition E−E₀ ∝ T（每模 k_BT），
            # 用 E = E₀ + c·T 线性拟合最低 4 个 T（T² 拟合系统性高估 E₀，实测 +0.4~0.8 meV）
            n_ex = min(4, nT)
            Tf = T_s[-n_ex:]
            Uf = U_s[-n_ex:, j]
            if len(Tf) < 2:
                E0 = Uf[0]
            else:
                _, E0 = np.polyfit(Tf, Uf, 1)
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

        # ---- 平衡自洽校验：热力学恒等式 C = d⟨E⟩/dT ----
        # 冷却/加热扫描平衡不足（滞后）或慢模式采样不足时 E(T) 斜率/涨落热容
        # 显著背离（2026-08-14 实测：短平衡 47×，长平衡仍 ~6-12× 来自 FM 团簇慢模式）——
        # S_abs 与等熵 ΔT_ad 的不确定性随该比值增大。显式提示而非静默输出。
        dUdT = np.abs(np.diff(U, axis=0)) / np.abs(np.diff(T_list))[:, None]  # meV/K
        C_slope = dUdT / KB_MEV_PER_K                                        # k_B/spin
        ratio = C_slope / np.maximum(C[:-1, :], 1e-9)
        if ratio.max() > 10.0:
            print(f"⚠️ 平衡提示：能量斜率热容/涨落热容最大比 {ratio.max():.1f}× "
                  f"（平衡不足或慢模式未采样 → S_abs/等熵 ΔT_ad 不确定性大；"
                  f"建议增大 equip_steps/calc_steps 或与加热路径对照）")
        elif ratio.max() > 3.0:
            print(f"ℹ️ 平衡提示：能量斜率热容/涨落热容最大比 {ratio.max():.1f}× "
                  f"（慢模式贡献；S_abs/ΔT_ad 存在 ~k_B/自旋级不确定性）")

        # ---- ΔS_M：麦克斯韦关系 ΔS_M(T) = ∫ (∂M/∂T)_B dB ----
        # 比绝对熵积分稳：无 E₀ 外推、无 β 尾巴；T→0 时 ∂M/∂T→0 自动归零。
        # ⚠️ B=0 列必须取 |M|：无场时自发磁化方向随机（单 seed 冷却扫描会在
        # 相变处从 + 翻转到 −），∂M_z/∂T 产生虚假正峰 → ΔS_M(0→2T) 低温变正
        # （"反铁磁假象"）。|M_z| 是场方向无关的自发磁化标量，物理正确
        # （2026-08-14 用户质疑"20-30K 反铁磁"定位到此根因）。
        M_eff = M.copy()
        M_eff[:, 0] = np.abs(M_eff[:, 0])
        # T 升序排列后中心差分，再做 3 点滑动平均压噪声。
        T_asc = np.argsort(T_list)
        M_asc = M_eff[T_asc]                  # (nT, nB)，T 升序
        T_a = T_list[T_asc]
        dM_dT = np.zeros_like(M_asc)
        dM_dT[1:-1] = (M_asc[2:] - M_asc[:-2]) / (T_a[2:] - T_a[:-2])[:, None]
        dM_dT[0] = (M_asc[1] - M_asc[0]) / (T_a[1] - T_a[0])
        dM_dT[-1] = (M_asc[-1] - M_asc[-2]) / (T_a[-1] - T_a[-2])
        # 5 点加权滑动平均 [1,4,6,4,1]/16（边界 clamp 复制端点）
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
        # Δs/k_B = (μ_s/k_B)·∫(∂m/∂T)dB（m 为无量纲每自旋平均、B 用 Tesla；
        # 缺 μ_s/k_B≈1.343 因子结果系统性小 0.744×——J=0 解析裁决实测）
        dS_M *= MU_S_MEV_PER_T / KB_MEV_PER_K

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

        # ---- ΔT_ad 质量门（2026-08-14 长平衡实测后加入）----
        # ① 信号阈值：|ΔS_M| < 10% 峰值 → 等熵解是数值噪声（高温段慢模式/统计涨落
        #    使 S 场差与斜率不可判定——T>1.5Tc 处典型）
        dS_scale = np.max(np.abs(dS_M), axis=0, keepdims=True)
        dT_ad[np.abs(dS_M) < 0.1 * dS_scale] = np.nan
        # ② 一致性：S_abs 场差 vs 麦克斯韦 ΔS_M 分歧 > 100% → 等熵输入不可信
        #    （低温段 E(T) 滞后残余导致 S 差虚大）
        ds_abs = S[:, 1:] - S[:, :1]                      # 同 dS_M 列语义（conv 单位）
        incons = np.abs(ds_abs - dS_M) / np.maximum(np.abs(dS_M), 1e-12)
        dT_ad[incons > 1.0] = np.nan

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

        atoms = []      # (frac_x, frac_y, frac_z, spin, species)
        for k, (x, y) in enumerate(sites):
            for b in range(Nb):
                cart2 = coords_all[x, y, b]         # (2,)
                frac = inv @ cart2
                frac = frac % 1.0
                frac = np.where(frac > 1.0 - 1e-10, 0.0, frac)   # 浮点舍入保护
                spin = cs[k // cs.shape[1], k % cs.shape[1], b] * spin_scale
                atoms.append((frac[0], frac[1], 0.0, spin, None))
        n_atoms = len(atoms)

        if species is None:
            sp_list = ["Spin"] * n_atoms
        else:
            sp_list = list(species)
            if len(sp_list) != n_atoms:
                raise ValueError(f"species 长度 {len(sp_list)} != 原子数 {n_atoms}")
        # 按物种分组重排（POSCAR 要求：原子按物种行顺序排列，第 7 行为逐物种计数）
        sp_order = sorted(set(sp_list))
        counts = [sp_list.count(s) for s in sp_order]
        order = [i for s in sp_order for i in range(n_atoms) if sp_list[i] == s]
        atoms = [atoms[i] + (sp_list[i],) for i in order]

        lines = [f"Magnetic cell from Monte Carlo (order={r['order']}, Q={r['top_charge']:.3f}, "
                 f"cell={r['cell_matrix']})"]
        lines.append("1.0")
        for L in (L1, L2, L3):
            lines.append(f"  {L[0]: .6f} {L[1]: .6f} {L[2]: .6f}")
        lines.append(" ".join(sp_order))
        lines.append(" ".join(map(str, counts)))
        lines.append("Direct")
        for (fx, fy, fz, _s, _sp) in atoms:
            lines.append(f"  {fx: .8f} {fy: .8f} {fz: .8f}")

        magmom = " ".join(f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}" for (_f, _g, _h, s, _sp) in atoms)

        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w") as f:
                f.write(text)
            if write_magmom:
                with open(path + ".magmom", "w") as f:
                    f.write(f"MAGMOM = {magmom}\n")
        return text, magmom

    def skyrmion_topological_density(self):
        """每格点拓扑荷密度 ρ_Q(x,y)（Berg–Lüscher 三角形贡献 1/3 分摊到三顶点）。

        ∫ρ_Q d²r = Q_total（skyrmion 数，反 skyrmion 为负）。中心检测见
        skyrmion_positions（mz 局部极小 + 拓扑荷验证）。仅支持 Nb=1 三角格。
        """
        if self.lat.Nb != 1:
            raise NotImplementedError("拓扑荷密度目前只适用于 Nb=1 的三角晶格")
        Nx, Ny = self.lat.Nx, self.lat.Ny

        def solid_angle(a, b, c):
            num = np.dot(a, np.cross(b, c))
            den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
            return 2.0 * np.arctan2(num, den)

        rho = np.zeros((Nx, Ny))
        for x in range(Nx):
            xp = (x + 1) % Nx
            for y in range(Ny):
                yp = (y + 1) % Ny
                s00 = self.spins[x, y, 0]
                s10 = self.spins[xp, y, 0]
                s01 = self.spins[x, yp, 0]
                s11 = self.spins[xp, yp, 0]
                w = solid_angle(s00, s10, s01) / (4.0 * np.pi) / 3.0
                rho[x, y] += w
                rho[xp, y] += w
                rho[x, yp] += w
                w = solid_angle(s10, s11, s01) / (4.0 * np.pi) / 3.0
                rho[xp, y] += w
                rho[xp, yp] += w
                rho[x, yp] += w
        return rho

    def skyrmion_positions(self, min_density=0.25, mz_core=-0.5, r_verify=2):
        """skyrmion 中心检测：mz 局部极小 + 拓扑荷验证。

        1. 候选 = mz < mz_core 的 3×3 邻域极小
        2. 验证：候选周围 r_verify 内 |Σρ_Q| ≥ min_density（真 skyrmion）
        3. 邻近候选去重（取 ρ_Q 绝对值更大者）

        返回 [(x, y, Q_local), ...]。
        """
        rho = self.skyrmion_topological_density()
        Nx, Ny = rho.shape
        mz = self.spins[:, :, 0, 2]
        cands = []
        for x in range(Nx):
            for y in range(Ny):
                if mz[x, y] >= mz_core:
                    continue
                if all(mz[x, y] <= mz[(x+dx) % Nx, (y+dy) % Ny]
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       if not (dx == 0 and dy == 0)):
                    cands.append((x, y))
        # 拓扑荷验证 + 去重（合并半径 r_verify 内候选；PBC 最小镜像距离，
        # 避免跨边界候选漏合并导致双计）
        centers = []
        for (x, y) in sorted(cands, key=lambda c: mz[c]):
            if any(min((x-cx) % Nx, (cx-x) % Nx) <= r_verify
                   and min((y-cy) % Ny, (cy-y) % Ny) <= r_verify
                   for (cx, cy, _q) in centers):
                continue
            # 局部拓扑荷
            q_local = 0.0
            for dx in range(-r_verify, r_verify+1):
                for dy in range(-r_verify, r_verify+1):
                    q_local += rho[(x+dx) % Nx, (y+dy) % Ny]
            if abs(q_local) >= min_density:
                centers.append((x, y, q_local))
        centers.sort(key=lambda c: -abs(c[2]))
        return centers

    def skyrmion_statistics(self, mz_cross=0.0):
        """skyrmion 统计：半径（mz=mz_cross 等值面）、面积占比、密度、晶格常数。

        - 半径：从中心沿 6 个 NN 方向找 mz 首次过 mz_cross 的距离，取平均
        - 密度：N_skyrmion / 超胞面积（以 |a1×a2| 为单位）
        - 晶格常数：中心最近邻距离（无 PBC 距离的近似——用最小非零距离）
        """
        centers = self.skyrmion_positions()
        Nx, Ny = self.lat.Nx, self.lat.Ny
        coords = self.lat.get_cartesian_coords()
        a1, a2 = np.array(self.lat.a_vecs[0]), np.array(self.lat.a_vecs[1])
        cell_area = abs(a1[0]*a2[1] - a1[1]*a2[0])

        radii, qs = [], []
        for (x, y, q) in centers:
            # 6 个 NN 方向（三角格）
            dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            rs = []
            for (dx, dy) in dirs:
                # 从中心向外扫描 mz 剖面
                for t in range(1, max(Nx, Ny)):
                    x2 = (x + t*dx) % Nx
                    y2 = (y + t*dy) % Ny
                    mz = self.spins[x2, y2, 0, 2]
                    r_now = np.hypot(t*dx*a1[0] + t*dy*a2[0],
                                     t*dx*a1[1] + t*dy*a2[1])
                    if mz >= mz_cross:
                        # 线性插值细化
                        if t > 1:
                            x1 = (x + (t-1)*dx) % Nx
                            y1 = (y + (t-1)*dy) % Ny
                            mz0 = self.spins[x1, y1, 0, 2]
                            r_prev = np.hypot((t-1)*dx*a1[0] + (t-1)*dy*a2[0],
                                              (t-1)*dx*a1[1] + (t-1)*dy*a2[1])
                            if mz0 != mz:
                                f = (mz_cross - mz0) / (mz - mz0)
                                r_now = r_prev + f * (r_now - r_prev)
                        rs.append(r_now)
                        break
                else:
                    rs.append(None)
            ok = [r for r in rs if r is not None]
            if ok:
                radii.append(np.mean(ok))
            qs.append(q)

        # 晶格常数：中心间最近距离（周期性最小镜像）
        lat_const = None
        if len(centers) >= 2:
            from itertools import combinations
            dmin = 1e18
            pts = [np.array([c[0], c[1]]) for c in centers]
            for i, j in combinations(range(len(pts)), 2):
                d = pts[i] - pts[j]
                for sx in (-1, 0, 1):
                    for sy in (-1, 0, 1):
                        dv = d + np.array([sx*Nx, sy*Ny])
                        rv = dv[0]*a1 + dv[1]*a2
                        dmin = min(dmin, np.hypot(rv[0], rv[1]))
            lat_const = dmin if dmin < 1e17 else None

        return {
            "n_skyrmions": len(centers),
            "centers": centers,
            "radius_mean": float(np.mean(radii)) if radii else None,
            "radius_std": float(np.std(radii)) if radii else None,
            "total_Q": float(np.sum(rho := self.skyrmion_topological_density())),
            "density_per_area": len(centers) / (Nx*Ny*cell_area) if cell_area else None,
            "lattice_constant": lat_const,
        }

    def skyrmion_stability(self, T_list, equip_steps, calc_steps, output_file=None):
        """升温扫描：skyrmion 存活数 vs 温度（热稳定性曲线）。

        从当前构型开始，逐 T 升温：每 T 先平衡（equip_steps sweeps）再统计
        （calc_steps 平均）skyrmion 数量（局部极大个数，min_density 用其 0.25）。

        返回 (T_list, N_list, N_std_list)。
        """
        out = []
        for T in T_list:
            for _ in range(equip_steps):
                self.mc_step(float(T))
            ns = []
            for _ in range(calc_steps):
                self.mc_step(float(T))
                ns.append(len(self.skyrmion_positions()))
            mean, std = float(np.mean(ns)), float(np.std(ns))
            out.append((T, mean, std))
            print(f"T = {T:7.3f} K | N_skyrmion = {mean:6.2f} ± {std:5.2f}")
        if output_file:
            with open(output_file, "w") as f:
                f.write("# T(K) N_skyrmion N_std\n")
                for (T, m, s) in out:
                    f.write(f"{T:.4f} {m:.4f} {s:.4f}\n")
        return ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out])

    def run_phase_diagram(self, B_list, T_list, equip_steps, calc_steps,
                          sample_interval=5, protocol="cooling",
                          classify=True, verbose=True, output_file=None):
        """B-T 磁相图扫描：每 (B, T) 点平衡 → 分类磁结构。

        Parameters
        ----------
        B_list : 磁场列表（Tesla，标量沿 z），外循环。
        T_list : 温度列表（K），内循环。
        equip_steps / calc_steps : 每点平衡/统计 sweeps。
        protocol :
            'cooling'   — field-cooled：每个 B 从最高 T 起，逐 T 降温，
                          上一点构型 warm start（退火路径，skyrmion 相最易出现）。
            'heating'   — 每个 B 从最低 T 起逐 T 升温（ZFC-like，检查滞后）。
            'fresh'     — 每点独立随机态（无记忆，最慢，无滞后路径）。
        classify : True → 每点跑 magnetic_structure_analysis + skyrmion 计数。
        output_file : 可选，写 CSV（B,T,phase,M,Q,n_sk）。

        Returns
        -------
        dict:
            phases : (len(B), len(T)) 相标签数组（'FM'|'AFM'|'helical'|
                     'skyrmion_lattice'|'multi-q'|'PM'）
            M, Q, n_sk : 同形状浮点数组
            B_list, T_list : 输入网格
        """
        if protocol not in ("cooling", "heating", "fresh"):
            raise ValueError("protocol 必须为 'cooling' / 'heating' / 'fresh'")
        if len(B_list) == 0 or len(T_list) == 0:
            raise ValueError("B_list / T_list 不能为空")
        if sample_interval <= 0:
            raise ValueError("sample_interval 必须 > 0")
        phases = np.empty((len(B_list), len(T_list)), dtype=object)
        M_arr = np.zeros((len(B_list), len(T_list)))
        Q_arr = np.zeros((len(B_list), len(T_list)))
        nsk_arr = np.zeros((len(B_list), len(T_list)))
        nb1 = self.lat.Nb == 1          # 拓扑荷/定位仅 Nb=1 支持（多子格跳过，防 NotImplementedError）

        # 协议显式排序（输入 T_list 任意顺序均可）：cooling=降温、heating=升温、
        # fresh=每点独立随机（顺序无关）
        T_asc = sorted(float(t) for t in T_list)
        T_order = list(reversed(T_asc)) if protocol == "cooling" else T_asc
        it_of = {T: i for i, T in enumerate(T_list)}

        orig_B = self.ham.B_field_meV.copy()
        try:
            for ib, B in enumerate(B_list):
                self.ham.B_field_meV = np.array([0.0, 0.0, B]) * MU_S_MEV_PER_T
                if protocol == "fresh":
                    self.spins = self._random_spins()
                for T in T_order:
                    it = it_of[T]
                    for _ in range(equip_steps):
                        self.mc_step(float(T))
                    # 统计平均（n_hits 显式计数采样次数）
                    Msum = Qsum = nsk = 0.0
                    n_samp = 0
                    n_hits = 0
                    for _ in range(calc_steps):
                        self.mc_step(float(T))
                        Msum += self.get_magnetization()[0]
                        if n_samp % sample_interval == 0:
                            n_hits += 1
                            if nb1:
                                Qsum += abs(self.topological_charge())
                                if classify:
                                    nsk += len(self.skyrmion_positions())
                        n_samp += 1
                    M_arr[ib, it] = Msum / n_samp
                    if n_hits:
                        Q_arr[ib, it] = Qsum / n_hits
                    if classify:
                        res = self.magnetic_structure_analysis()
                        order = res["order"]
                        n_sk = nsk / n_hits if n_hits else 0.0
                        nsk_arr[ib, it] = n_sk
                        # skyrmion 计数优先于 S(q) 分类（S(q) 对少 skyrmion 不敏感）
                        if nb1 and n_sk >= 1 and res["top_charge"] != 0:
                            order = "skyrmion_lattice"
                        phases[ib, it] = order
                        if verbose:
                            print(f"B={B:5.2f} T T={T:7.3f} K | {order:16s} | "
                                  f"M={M_arr[ib, it]:.3f} "
                                  f"Q={Q_arr[ib, it]:.2f} "
                                  f"N_sk={n_sk:.1f}")
                    else:
                        if verbose:
                            print(f"B={B:5.2f} T T={T:7.3f} K | M={M_arr[ib, it]:.3f}")
        finally:
            self.ham.B_field_meV = orig_B
        if output_file:
            with open(output_file, "w") as f:
                f.write("B,T,phase,M,Q,n_skyrmion\n")
                for ib, B in enumerate(B_list):
                    for it, T in enumerate(T_list):
                        f.write(f"{B:.4f},{T:.4f},{phases[ib, it]},"
                                f"{M_arr[ib, it]:.6f},{Q_arr[ib, it]:.6f},{nsk_arr[ib, it]:.4f}\n")
        return {"phases": phases, "M": M_arr, "Q": Q_arr, "n_sk": nsk_arr,
                "B_list": np.asarray(B_list), "T_list": np.asarray(T_list)}

    def run_spin_dynamics(self, dt, n_steps, T, damping=0.1, save_interval=10,
                          seed=None, output_prefix=None):
        """Langevin 自旋动力学（经典 LLG + 热噪声，Heun 积分）。

        运动方程（原子单位制，每步时间 dt·ℏ/J）：
          ∂S_i/∂t = −γ' S_i × (H_eff,i + ξ_i) − λ S_i × (S_i × H_eff,i)
          H_eff,i = −∂H/∂S_i（局域有效场，含交换/DMI/各向异性/外场）
          热噪声 <ξ_iα(t) ξ_jβ(t')> = 2λ k_B T δ_ij δ_αβ δ(t−t')

        用 Heun（二阶随机 Runge–Kutta）积分；每个保存间隔记录一帧自旋构型。
        返回轨迹 spins_t (n_frames, Nx, Ny, Nb, 3) + 时间数组。
        """
        Nx, Ny, Nb, _ = self.spins.shape
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self._rng
        lam = damping
        # 有效场（局域能量梯度，符号：H = −Σ S_i·h_i → h_i = −∂H/∂S_i）
        # 用能量差分或解析梯度：这里用解析梯度（键展开）

        def local_field():
            """H_eff = −∂H/∂S：交换 −ΣJ·S_j（同子格键正反两条目在本列表各 0.5；
            跨子格/on-site 键仅一条目，全强度——统一 0.5 会砍半跨子格场），
            各向异性 −2A·S_z ẑ，外场 +B。"""
            h = np.zeros_like(self.spins)
            for b in range(Nb):
                for (nu, dx, dy, Jm) in self.ham.bonds[b]:
                    # S(r+δ)：roll 负向 → spins[r-δ] 移位后位置 r 存 S(r+δ)
                    S_sh = np.roll(self.spins, (-dx, -dy), axis=(0, 1))[:, :, nu]
                    f = 0.5 if (nu == b and (dx != 0 or dy != 0)) else 1.0
                    h[:, :, b] -= f * (S_sh @ Jm.T)
            for b in range(Nb):
                h[:, :, b, 2] -= 2.0 * self.ham.A[b] * self.spins[:, :, b, 2]
                h[:, :, b] += self.ham.B_field_meV
            return h

        def heun_step(S0, h, dt, T, lam):
            """Heun 一步。S0 可能与 self.spins 同一引用——必须先拷贝，
            否则 self.spins[:] = S1 会污染 S0，导致 k1 与 S1 不匹配（历史爆炸根因）。
            噪声幅度 √(2λk_BT/dt)（T 需乘 k_B 换 meV）；校正步复用同一 Wiener
            增量（标准随机 Heun——重新抽样 xi1 破坏涨落-耗散平衡）。"""
            S = S0.copy()
            # 热噪声场（k_B·T 才是能量单位）
            xi = rng.standard_normal(S.shape) * np.sqrt(2.0 * lam * KB_MEV_PER_K * T / dt)
            # k1 = dS/dt
            SxH = np.cross(S, h + xi)
            k1 = -SxH - lam * np.cross(S, SxH)
            S1 = S + dt * k1
            S1 = S1 / np.linalg.norm(S1, axis=-1, keepdims=True)
            # 重算场（同一噪声增量）
            self.spins[:] = S1
            h1 = local_field()
            SxH1 = np.cross(S1, h1 + xi)
            k2 = -SxH1 - lam * np.cross(S1, SxH1)
            S_new = S + 0.5 * dt * (k1 + k2)
            S_new = S_new / np.linalg.norm(S_new, axis=-1, keepdims=True)
            return S_new

        frames, times = [], []
        for step in range(n_steps):
            h = local_field()
            self.spins[:] = heun_step(self.spins, h, dt, T, lam)
            if step % save_interval == 0:
                frames.append(self.spins.copy())
                times.append(step * dt)
        if output_prefix:
            arr = np.stack(frames)
            np.save(f"{output_prefix}_traj.npy", arr)
            np.savetxt(f"{output_prefix}_times.txt", np.array(times))
        return np.stack(frames), np.array(times)

    def skyrmion_diffusion(self, traj, times, match_cutoff=3.0, min_track=2):
        """从动力学轨迹提取 skyrmion 质心轨迹 → MSD → 扩散系数 D。

        Parameters
        ----------
        traj : (n_frames, Nx, Ny, Nb, 3) 动力学轨迹（run_spin_dynamics 返回）。
        times : (n_frames,) 时间数组（相同时间单位）。
        match_cutoff : 相邻帧中心配对的最大位移（格点单位，PBC 最小镜像）。
        min_track : 参与 MSD 的最短存活帧数（排除瞬态噪声）。

        Returns
        -------
        dict:
            tracks   : 每条 skyrmion 轨迹 (n_track, 2) 格点坐标列表（存活帧序列）
            msd      : MSD(τ) 数组（对 τ 的所有起点平均）
            tau      : 对应 τ 数组
            D, D_err : 扩散系数（格点²/时间单位）与拟合误差（线性拟合前 1/3 τ）
            n_tracks : 追踪到的轨迹数
        """
        Nx, Ny = self.lat.Nx, self.lat.Ny
        centers = []
        for f in range(len(traj)):
            self.spins[:] = traj[f]
            pos = self.skyrmion_positions()
            centers.append(pos)
        # 相邻帧配对（PBC 最小镜像差），坐标逐步 unwrap——避免长 τ MSD 被
        # 最小镜像截断在 (N/2)² 导致扩散系数 D 系统性低估
        tracks = []          # 每条 = [(frame, x_unwrapped, y_unwrapped), ...]
        for f in range(1, len(centers)):
            prev = centers[f - 1]
            curr = centers[f]
            used = [False] * len(curr)
            for pi, (x0, y0, _q0) in enumerate(prev):
                best, bj, bd = None, -1, 1e9
                for j, (x1, y1, _q1) in enumerate(curr):
                    if used[j]:
                        continue
                    dx = min(abs(x1 - x0), Nx - abs(x1 - x0))
                    dy = min(abs(y1 - y0), Ny - abs(y1 - y0))
                    d = np.hypot(dx, dy)
                    if d < bd:
                        best, bj, bd = (x1, y1), j, d
                if best is not None and bd <= match_cutoff:
                    used[bj] = True
                    # 接到已有轨迹或开新轨迹（按上一帧坐标匹配，坐标已 unwrap）
                    hit = None
                    for tr in tracks:
                        if tr[-1][0] == f - 1 and tr[-1][1] % Nx == x0 and tr[-1][2] % Ny == y0:
                            hit = tr
                            break
                    if hit is None:
                        hit = [(f - 1, float(x0), float(y0))]
                        tracks.append(hit)
                    # unwrap：上一位置 + 最小镜像位移
                    dx = best[0] - x0
                    dy = best[1] - y0
                    dx -= Nx * round(dx / Nx)
                    dy -= Ny * round(dy / Ny)
                    hit.append((f, hit[-1][1] + dx, hit[-1][2] + dy))
        tracks = [tr for tr in tracks if len(tr) >= min_track]
        # MSD(τ)：所有帧对（unwrap 坐标直接差，无截断）
        tau_max = min(len(times), 25)
        msd = np.zeros(tau_max)
        tau = np.zeros(tau_max)
        cnt = np.zeros(tau_max)
        for tr in tracks:
            for i in range(len(tr)):
                for j in range(i + 1, min(i + tau_max, len(tr))):
                    dtau = tr[j][0] - tr[i][0]
                    if dtau >= tau_max:
                        break
                    dx = tr[j][1] - tr[i][1]
                    dy = tr[j][2] - tr[i][2]
                    msd[dtau] += dx * dx + dy * dy
                    tau[dtau] = times[tr[j][0]] - times[tr[i][0]]
                    cnt[dtau] += 1
        m = cnt > 0
        msd = msd[m] / np.maximum(cnt[m], 1)
        tau = tau[m]
        # Einstein: MSD = 4Dt（2D）——用前 1/3 线性段拟合
        nfit = max(2, len(msd) // 3)
        if nfit >= 2 and len(msd) >= 2:
            A = np.polyfit(tau[:nfit], msd[:nfit], 1)
            D, D_err = A[0] / 4.0, 0.0
            resid = msd[:nfit] - np.polyval(A, tau[:nfit])
            D_err = np.sqrt(np.sum(resid ** 2) / max(1, nfit - 2)) / 4.0
        else:
            D, D_err = 0.0, 0.0
        return {"tracks": tracks, "msd": msd, "tau": tau, "D": float(D),
                "D_err": float(D_err), "n_tracks": len(tracks)}

    def skyrmion_lifetime(self, traj, times, min_n=2):
        """skyrmion 存活数 N(t) → 指数衰减拟合 → 寿命 τ。

        对轨迹逐帧统计 skyrmion 数（用局部拓扑荷验证的核心计数），
        拟合 N(t) = N0·exp(−t/τ)（log 线性最小二乘）。

        Returns
        -------
        (t, N, N0, tau, R2) : 时间数组、存活数数组、初始数、寿命、拟合优度。
        """
        N = []
        for f in range(len(traj)):
            self.spins[:] = traj[f]
            N.append(len(self.skyrmion_positions()))
        t = np.asarray(times)
        N = np.asarray(N, dtype=float)
        mask = N > 0
        if mask.sum() < 3:
            return t, N, N[0], np.inf, 0.0
        # log 线性拟合（N 的噪声在指数尺度上）
        A = np.polyfit(t[mask], np.log(N[mask]), 1)
        tau = -1.0 / A[0] if A[0] < 0 else np.inf
        N0 = np.exp(A[1])
        pred = N0 * np.exp(-t / tau) if np.isfinite(tau) else np.full_like(t, N0, dtype=float)
        ss_res = np.sum((np.log(N[mask]) - np.log(pred[mask])) ** 2)
        ss_tot = np.sum((np.log(N[mask]) - np.mean(np.log(N[mask]))) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return t, N, N0, float(tau), float(R2)

    def arrhenius_analysis(self, T_list, tau_list, output_file=None):
        """Arrhenius 分析：ln τ vs 1/T → 湮灭势垒 E_b（meV）。

        τ(T) = τ0·exp(E_b / k_B T) → ln τ = E_b/(k_B·T) + ln τ0。
        返回 (E_b, ln_tau0, R2)。
        """
        T = np.asarray(T_list, dtype=float)
        tau = np.asarray(tau_list, dtype=float)
        fin = np.isfinite(tau) & (tau > 0)
        if fin.sum() < 2:
            raise ValueError("至少需要 2 个有限寿命数据点")
        kB = 0.0861733  # meV/K
        x, y = 1.0 / T[fin], np.log(tau[fin])
        A = np.polyfit(x, y, 1)
        E_b = A[0] * kB  # 斜率 = E_b/k_B → meV
        pred = np.polyval(A, x)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if output_file:
            with open(output_file, "w") as f:
                f.write("# T(K) tau tau_fit\n")
                for ti, tauv in zip(T[fin], tau[fin]):
                    f.write(f"{ti:.4f} {tauv:.6e} {np.exp(np.polyval(A, 1.0/ti)):.6e}\n")
        return {"E_b_meV": float(E_b), "ln_tau0": float(A[1]), "R2": float(R2)}

    def run_parallel_tempering(self, T_list, equip_steps, swap_interval, n_swaps,
                               B_field=(0.0, 0.0, 0.0), seed=None, verbose=True):
        """Parallel Tempering（副本交换 MC）——克服亚稳态/势垒卡滞。

        标准副本交换协议：
          1. n_replicas 个副本，温度 T_list（升序），每副本独立 RNG + 独立 spins；
          2. 本地演化：每副本跑 swap_interval 个 sweeps（Metropolis）；
          3. 相邻副本 (i, i+1) 交换尝试：
                 p = min(1, exp[(β_i − β_{i+1})(E_i − E_{i+1})]),
                 β = 1/(k_B T)
             接受则交换两副本的自旋构型；
          4. 循环 n_swaps 轮。

        高温暖副本翻越势垒 → 交换把高温构型“注入”低温副本，
        低温副本因此能访问亚稳态以外的构型（如 skyrmion 相）。

        Parameters
        ----------
        T_list : 副本温度（K，升序，最少 2 个）。
        equip_steps : 交换前每副本本地平衡 sweeps。
        swap_interval : 两次交换尝试之间的本地 sweeps。
        n_swaps : 交换尝试轮数。
        B_field : 外场（Tesla），应用到所有副本。
        seed : 随机种子（各副本 RNG 从 seed+i 派生）。

        Returns
        -------
        dict:
            spins_final : (n_replicas, Nx, Ny, Nb, 3) 各副本最终构型
            E_series    : (n_replicas, n_swaps+1) 每轮交换前能量（meV）
            T_visits    : (n_replicas, n_swaps+1) 各副本实际访问温度
            acc_rate    : (n_replicas-1,) 相邻副本对交换接受率
            E_hist      : 最低温副本的最终能量（标量，用于与单链对照）
        """
        T_list = np.asarray(T_list, dtype=float)
        if len(T_list) < 2:
            raise ValueError("Parallel Tempering 至少需要 2 个副本温度")
        if not np.all(np.diff(T_list) > 0):
            raise ValueError("T_list 必须严格升序")
        kB = 0.0861733  # meV/K
        beta = 1.0 / (kB * T_list)          # 1/meV
        n_rep = len(T_list)
        Nx, Ny, Nb, _ = self.spins.shape
        base_seed = seed if seed is not None else (getattr(self, "_seed", None) or 0)
        rng_swap = np.random.default_rng(base_seed + 777)  # 交换决策专用 RNG

        # 克隆副本：共享 lattice/hamiltonian 定义，独立 spins + RNG
        reps = []
        for r in range(n_rep):
            mc = MonteCarlo2D(self.lat, self.ham, seed=(base_seed + 1000 * (r + 1)))
            mc.spins = self.spins.copy() if r == 0 else None
            reps.append(mc)
        # 副本 0 用当前构型，其余用随机态
        for r in range(1, n_rep):
            reps[r].spins = reps[r]._random_spins()
        for mc in reps:
            mc.ham.B_field_meV = np.array(B_field, dtype=float) * MU_S_MEV_PER_T

        # 预热
        for r in range(n_rep):
            for _ in range(equip_steps):
                reps[r].mc_step(T_list[r])

        E_series = np.zeros((n_rep, n_swaps + 1))
        T_visits = np.zeros((n_rep, n_swaps + 1))
        E_series[:, 0] = [reps[r].total_energy() for r in range(n_rep)]
        T_visits[:, 0] = T_list
        walker_T = list(T_list)              # walker r 当前实际温度（仅报告用）
        n_acc = np.zeros(n_rep - 1)

        # 固定温度槽表述：每副本在其槽温演化，交换仅交换构型；walker_T 只做
        # 报告——混合"walker 温演化 + 槽温判据"会破坏细致平衡
        for s in range(n_swaps):
            # 本地演化（各副本在其固定槽温）
            for r in range(n_rep):
                for _ in range(swap_interval):
                    reps[r].mc_step(T_list[r])
            # 相邻交换（奇偶交替；每对每轮恰好 1 次尝试）
            for parity in (0, 1):
                for i in range(parity, n_rep - 1, 2):
                    E_i, E_j = reps[i].total_energy(), reps[i + 1].total_energy()
                    delta = (beta[i] - beta[i + 1]) * (E_i - E_j)
                    if delta >= 0.0 or rng_swap.random() < np.exp(delta):
                        reps[i].spins, reps[i + 1].spins = reps[i + 1].spins, reps[i].spins
                        walker_T[i], walker_T[i + 1] = walker_T[i + 1], walker_T[i]
                        n_acc[i] += 1
            E_series[:, s + 1] = [reps[r].total_energy() for r in range(n_rep)]
            T_visits[:, s + 1] = walker_T
            if verbose and (s + 1) % max(1, n_swaps // 10) == 0:
                acc = n_acc / (s + 1.0)
                print(f"PT sweep {s+1}/{n_swaps} | acc: "
                      + " ".join(f"{a:.2f}" for a in acc))

        spins_final = np.stack([reps[r].spins for r in range(n_rep)])
        return {"spins_final": spins_final, "E_series": E_series,
                "T_visits": T_visits,
                "acc_rate": n_acc / n_swaps,
                "E_hist": E_series[0]}

    def dynamic_structure_factor(self, traj, times, q_grid=None, t_max=None):
        """动力学结构因子 S(q,ω)（经典，从自旋轨迹）。

        S(q,ω) = (1/2π)∫dt e^{iωt} ⟨S_q(t)·S_{−q}(0)⟩

        traj  : (n_frames, Nx, Ny, Nb, 3) 动力学轨迹（run_spin_dynamics 输出）
        times : 帧时间数组
        q_grid: 采样 q 点（分数坐标列表）；None → 全网格降采样（stride 4）
        t_max : 相关时间截断（帧数）；None → 全轨迹

        返回 (q_list, omega, S) —— S[n_q, n_omega]。
        """
        n_frames, Nx, Ny, Nb, _ = traj.shape
        if t_max is not None:
            n_frames = min(n_frames, t_max)
        if q_grid is None:
            stride = max(1, Nx // 8)
            q_grid = [(i, j) for i in range(0, Nx, stride) for j in range(0, Ny, stride)]
        dt = times[1] - times[0] if len(times) > 1 else 1.0

        # S_q(t) = Σ_b e^{−2πi(q·τ_b)} Σ_r S_b(r,t) e^{−2πi q·r}（q 为整数索引，
        # 子格相位在频域、跨子格相干求和——与 spin_structure_factor 同约定）
        Sqt = np.zeros((len(q_grid), n_frames, 3), dtype=complex)
        xg = np.arange(Nx)[:, None]
        yg = np.arange(Ny)[None, :]
        for a, (qx, qy) in enumerate(q_grid):
            ph = np.exp(-2j*np.pi*(qx*xg/Nx + qy*yg/Ny))
            for t in range(n_frames):
                for comp in range(3):
                    acc = 0j
                    for b in range(Nb):
                        f1, f2 = self.lat.basis[b]
                        acc += (np.sum(traj[t, :, :, b, comp] * ph)
                                * np.exp(-2j*np.pi*(qx*f1/Nx + qy*f2/Ny)))
                    Sqt[a, t, comp] = acc

        # 线性自相关 C(τ) = ⟨S_q(t+τ)·S_{−q}(t)⟩（FFT 零填充避免循环环绕，
        # 每 τ 用有效样本数 n_frames−τ 归一）
        n_f = n_frames // 2
        L = 2 * n_frames
        norm = n_frames - np.arange(n_f, dtype=float)
        C = np.zeros((len(q_grid), n_f), dtype=complex)
        for a in range(len(q_grid)):
            for comp in range(3):
                x = Sqt[a, :, comp]
                ac = np.fft.ifft(np.fft.fft(x, L) * np.conj(np.fft.fft(x, L)))[:n_f].real
                C[a] += ac / norm

        # 时间 → 频率（单边谱，含因子 2）：S(q,ν) = [2·Re Σ_{τ≥0} C(τ)e^{−i2πντ} − C(0)]·dt
        omega = np.fft.rfftfreq(n_f, d=dt)     # 循环频率 ν = ω/2π
        S = np.zeros((len(q_grid), len(omega)))
        for a in range(len(q_grid)):
            S[a] = (2.0 * np.fft.rfft(C[a].real).real - C[a, 0].real) * dt
        return q_grid, omega, S

    def sqw_peak_extraction(self, S, omega, q_grid, q_indices=None, fit="lorentzian",
                            verbose=True):
        """S(q,ω) 峰位 + 线宽提取（磁振子色散与阻尼的直接可观测量）。

        对每个选定 q：在 ω>0 区间定位谱峰（3 点局部极大），
        用 Lorentzian（磁振子阻尼线型）或 Gaussian 拟合：
            S(ω) = A·Γ² / ((ω−ω₀)² + Γ²)      [Lorentzian, FWHM = 2Γ]
            S(ω) = A·exp(−(ω−ω₀)²/(2σ²))      [Gaussian,   FWHM = 2√(2ln2)·σ]
        基线：S(ω) 最低 20% 分位中位数。

        Parameters
        ----------
        S : (n_q, n_ω) 动态结构因子（dynamic_structure_factor 输出）。
        omega : (n_ω,) 频率轴（时间单位⁻¹）。
        q_grid : q 点列表（dynamic_structure_factor 返回的整数索引）。
        q_indices : 要提取的 q 索引列表；None → 全部。
        fit : 'lorentzian'（默认）| 'gaussian'。
        verbose : 打印每 q 提取结果。

        Returns
        -------
        dict:
            q_frac     : (n,) 分数坐标 q = (qx/Nx, qy/Ny)
            omega_peak : 峰位（ω 单位）
            FWHM       : 线宽（ω 单位）
            amp        : 峰高 A
            R2         : 拟合优度
        """
        from scipy.optimize import curve_fit

        if q_indices is None:
            q_indices = list(range(len(q_grid)))
        out = {"q_frac": [], "omega_peak": [], "FWHM": [], "amp": [], "R2": []}
        pos_omega = omega > 0
        w = omega[pos_omega]

        for qi in q_indices:
            sp = S[qi][pos_omega]
            if sp.size < 5:
                continue
            base = np.median(np.sort(sp)[: max(1, sp.size // 5)])
            sp = sp - base
            # 峰定位：3 点局部极大
            peak_i = -1
            peak_v = -1.0
            for i in range(1, sp.size - 1):
                if sp[i] > sp[i - 1] and sp[i] >= sp[i + 1] and sp[i] > peak_v:
                    peak_v = sp[i]
                    peak_i = i
            if peak_i < 0:
                continue
            w0 = w[peak_i]
            # 半高半宽初值
            half = peak_v / 2.0
            hw = 0.0
            for i in range(peak_i, sp.size):
                if sp[i] <= half:
                    hw = w[i] - w0
                    break
            hw = max(hw, w[1] - w[0])
            if fit == "lorentzian":
                def model(x, A, wc, G):
                    return A * G * G / ((x - wc) ** 2 + G * G)
                p0 = [peak_v, w0, hw]
                bounds = ([0, w[0], 1e-4], [np.inf, w[-1], 10 * (w[-1] - w[0])])
            else:
                def model(x, A, wc, s):
                    return A * np.exp(-(x - wc) ** 2 / (2 * s * s))
                p0 = [peak_v, w0, hw / 2.3548]
                bounds = ([0, w[0], 1e-4], [np.inf, w[-1], 10 * (w[-1] - w[0])])
            try:
                popt, _ = curve_fit(model, w, sp, p0=p0, bounds=bounds, maxfev=20000)
                pred = model(w, *popt)
                ss_res = np.sum((sp - pred) ** 2)
                ss_tot = np.sum((sp - np.mean(sp)) ** 2)
                R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except Exception:
                popt = p0
                R2 = 0.0
            FWHM = 2.0 * abs(popt[2]) if fit == "lorentzian" else 2.3548 * abs(popt[2])
            qx, qy = q_grid[qi]
            out["q_frac"].append((qx / self.lat.Nx, qy / self.lat.Ny))
            out["omega_peak"].append(float(popt[1]))
            out["FWHM"].append(float(FWHM))
            out["amp"].append(float(popt[0]))
            out["R2"].append(float(R2))
            if verbose:
                print(f"q={out['q_frac'][-1]} | ω₀={popt[1]:.4f} "
                      f"FWHM={FWHM:.4f} A={popt[0]:.2f} R²={R2:.3f}")
        for k in out:
            out[k] = np.asarray(out[k])
        return out

    def export_xyz(self, path="spins.xyz", species=None, spin_scale=1.0, trajectory=None):
        """导出自旋构型为 .xyz（原子 + 磁矩矢量列），OVITO 直接可视化。

        - 单帧：当前 self.spins；多帧：传 trajectory (n_frames, Nx, Ny, Nb, 3)
        - 每原子 3 列 = 自旋矢量 × spin_scale（OVITO 可渲染为 vector 场）
        - species 可传元素符号列表（每原子一个，默认 'X'）
        """
        if trajectory is None:
            frames = [self.spins.copy()]
        else:
            frames = list(trajectory)
        Nx, Ny, Nb, _ = frames[0].shape
        coords = self.lat.get_cartesian_coords()
        n_atoms = Nx * Ny * Nb
        if species is None:
            sp = ["X"] * n_atoms
        elif isinstance(species, str):
            sp = [species] * n_atoms
        else:
            sp = list(species)
            if len(sp) != n_atoms:
                raise ValueError(f"species 长度 {len(sp)} != 原子数 {n_atoms}")

        lines = []
        for fi, S in enumerate(frames):
            lines.append(str(n_atoms))
            lines.append(f"Frame {fi}  Lattice=\"{self.lat.a_vecs[0][0]:.6f} {self.lat.a_vecs[0][1]:.6f} 0.0 "
                         f"{self.lat.a_vecs[1][0]:.6f} {self.lat.a_vecs[1][1]:.6f} 0.0 "
                         f"0.0 0.0 10.0\"  Properties=species:S:1:pos:R:3:spin:R:3")
            k = 0
            for x in range(Nx):
                for y in range(Ny):
                    for b in range(Nb):
                        rx, ry = coords[x, y, b]
                        s = S[x, y, b] * spin_scale
                        lines.append(f"{sp[k]} {rx:.6f} {ry:.6f} 0.0 {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}")
                        k += 1
        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

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

        # 显式设置外场，避免继承此前退火或磁滞计算留下的状态；结束恢复原值
        orig_B = self.ham.B_field_meV.copy()
        self.ham.B_field_meV = B_field * MU_S_MEV_PER_T
        results = []
        print(f"--- Numba: 温度扫描 (B={B_field} Tesla) ---")
        try:
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
        finally:
            self.ham.B_field_meV = orig_B

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

        orig_B = self.ham.B_field_meV.copy()
        try:
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
        finally:
            self.ham.B_field_meV = orig_B

        np.savetxt(output_file, results, header="B_z(T) M_z_mean_spin", fmt="%.8f")
        return np.asarray(results)

    def run_magnetization_curves(self, T_list, B_list, equip_steps, calc_steps,
                                 sample_interval=4, output_csv=None,
                                 protocol="cooling"):
        """等温磁化曲线 M(B) 与热磁曲线 M(T)：T×B 双重扫描采样 <M_z>。

        循环序 B 外 T 内（同 run_magnetocaloric）。protocol 决定每个 B 内的
        T 访问顺序（与输入顺序无关，结果仍按 T_list 原索引存放）：
          'cooling' — 每个场从最高 T 逐点降温（零场冷却后升场，FC 式，默认）
          'heating' — 从最低 T 逐点升温
        每个 (T,B) 平衡 ``equip_steps`` sweep 后对 ``calc_steps`` 个记录点求
        <M_z> 平均。返回 (T_list, B_list, M[nT, nB])：M[i,j] = M(T_i, B_j)
        （每自旋 z 分量）。output_csv 给定时写 3 列 T B M。
        """
        import numpy as np
        if protocol not in ("cooling", "heating"):
            raise ValueError("protocol 必须为 'cooling' / 'heating'")
        nT, nB = len(T_list), len(B_list)
        M = np.zeros((nT, nB))
        # 显式排序（同 run_phase_diagram 的协议处理）：cooling=降温、heating=升温
        T_asc = sorted(float(t) for t in T_list)
        T_order = list(reversed(T_asc)) if protocol == "cooling" else T_asc
        it_of = {T: i for i, T in enumerate(T_list)}
        print(f"--- 磁化曲线: {nT} 温度 × {nB} 场 (protocol={protocol}) ---")
        orig_B = self.ham.B_field_meV.copy()
        try:
            for j, Bz in enumerate(B_list):
                self.ham.B_field_meV = np.array([0.0, 0.0, Bz], dtype=np.float64) * MU_S_MEV_PER_T
                for T in T_order:
                    it = it_of[T]
                    self._validate_sampling(float(T), equip_steps, calc_steps, sample_interval)
                    for _ in range(equip_steps):
                        self.mc_step(T)
                    m_acc = 0.0
                    for _ in range(calc_steps):
                        for _ in range(sample_interval):
                            self.mc_step(T)
                        m_acc += self.get_magnetization()[1][2]
                    M[it, j] = m_acc / calc_steps
                    print(f"  B={Bz:5.2f} T  T={T:6.1f} K  M={M[it,j]:.3f}")
        finally:
            self.ham.B_field_meV = orig_B
        if output_csv:
            with open(output_csv, "w") as f:
                f.write("# T(K) B(T) M_per_spin\n")
                for i in range(nT):
                    for j in range(nB):
                        f.write(f"{T_list[i]:.4f} {B_list[j]:.4f} {M[i,j]:.8f}\n")
        return T_list, B_list, M


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
                   header=("T(K) M_mean T_std M_std Chi_perT_mean(3) "
                           "Chi_perT_std(3) Chi_permeV_mean(3) "
                           "Chi_permeV_std(3) C_kB_mean C_kB_std"),
                   fmt="%.8f")
    print("完成。mean±std 已保存。")
    return mean, std, all_res
