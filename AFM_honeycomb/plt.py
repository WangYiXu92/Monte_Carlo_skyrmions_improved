import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 模拟参数与绘图模式控制
# ==========================================
Nx, Ny = 120, 120   # 超胞大小
Nb = 2              # 原胞内的原子数

# --- 绘图控制 ---
show_quiver = True      # 是否显示箭头
show_contourf = False   # 是否显示背景云图 (Sz)
sparse_step = 1         # 箭头的稀疏步长 (对于 120x120, 建议 3-5)

# ==========================================
# 2. 读取并重组数据
# ==========================================
print("正在读取数据...")
raw_data = np.loadtxt("skyrmion_spins.txt", skiprows=1)
grid_data = raw_data.reshape((Nx, Ny, Nb, 5))

# ==========================================
# 3. 数据准备 (区分全量与稀疏)
# ==========================================
# A. Contourf 始终使用全量数据，保证背景平滑
Sz_full = grid_data[:, :, 0, 4] 
X_full = grid_data[:, :, 0, 0]
Y_full = grid_data[:, :, 0, 1]

# B. Quiver 使用稀疏化数据，防止箭头太密
sparse_grid = grid_data[::sparse_step, ::sparse_step, :, :]
sparse_flat = sparse_grid.reshape((-1, 5))

X_q = sparse_flat[:, 0]
Y_q = sparse_flat[:, 1]
Sx_q = sparse_flat[:, 2]
Sy_q = sparse_flat[:, 3]
Sz_q = sparse_flat[:, 4]

# ==========================================
# 4. 绘图设置
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# --- [A] 绘制背景填充图 (Contourf) ---
if show_contourf:
    # 使用 alpha 控制背景亮度，让彩色箭头更显眼
    cf = ax.contourf(X_full, Y_full, Sz_full, levels=50, cmap='coolwarm', vmin=-1, vmax=1, alpha=0.6)
    cbar_f = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar_f.set_label('Background $S_z$', fontsize=12)

# --- [B] 绘制矢量箭头 (Quiver) ---
if show_quiver:
    # 核心修正：
    # 这里的参数顺序是 (X, Y, U, V, C)
    # X, Y: 位置; Sx, Sy: 矢量方向; Sz_q: 用于映射颜色的数值
    
    # 宽度控制：0.001 左右会非常细
    q = ax.quiver(X_q, Y_q, Sx_q, Sy_q, Sz_q, 
                  cmap='coolwarm',     # 箭头颜色映射
                  clim=(-1.0, 1.0),    # 固定颜色范围
                  pivot='mid',         # 箭头中心对齐格点
                  scale=65,            # 箭头长度缩放（数值越大箭头越短）
                  width=0.0012,        # 箭身宽度 (调细)
                  headwidth=4,         # 箭头头部宽度 (相对于 width 的比例)
                  headlength=5)        # 箭头头部长度 (相对于 width 的比例)

    # 如果没有开启 contourf，则需要为 quiver 配一个颜色条
    if not show_contourf:
        cbar_q = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar_q.set_label('Spin $S_z$ (Vector Color)', fontsize=12)

# --- [C] 细节美化 ---
ax.set_aspect('equal') # 保证物理比例 1:1
ax.set_title(f"Skyrmion Configuration (Step={sparse_step})", fontsize=16)
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)

# ==========================================
# 5. 保存与显示
# ==========================================
plt.tight_layout()
out_name = f"skyrmion_plot_s{sparse_step}.png"
plt.savefig(out_name, dpi=300, bbox_inches='tight')
print(f"绘图完成！图片已保存为: {out_name}")
plt.show()
