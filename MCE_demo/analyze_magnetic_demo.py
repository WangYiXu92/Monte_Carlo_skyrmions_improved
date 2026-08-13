"""磁结构自动识别 demo：MC 退火后自动分类 + 提取磁单胞。

用法：python analyze_magnetic_demo.py
"""
import sys
import numpy as np
from mc import Lattice, Hamiltonian, MonteCarlo2D

# 三角格 skyrmion 参数（J=-40 meV, D=1.45, A=-0.04 易轴）
a = 1.0
a_vecs = [[a, 0.0], [0.5*a, np.sqrt(3.0)/2.0*a]]
lat = Lattice(a_vecs, [[0, 0]], 48, 48)
ham = Hamiltonian(Nb=1, A_ani=-0.04)
J_ex, D = -40.0, 1.45
ham.add_bond(0, 0, [1, 0],   [[J_ex,0,-D],[0,J_ex,0],[D,0,J_ex]])
ham.add_bond(0, 0, [0, 1],   [[J_ex,0,-0.5*D],[0,J_ex,-np.sqrt(3)/2*D],[0.5*D,np.sqrt(3)/2*D,J_ex]])
ham.add_bond(0, 0, [-1, 1],  [[J_ex,0,0.5*D],[0,J_ex,-np.sqrt(3)/2*D],[-0.5*D,np.sqrt(3)/2*D,J_ex]])
mc = MonteCarlo2D(lat, ham, seed=42)

mc.run_skyrmion_annealing(T_init=150.0, T_final=3.0, steps_per_T=600, B_field=0.15)

r = mc.magnetic_structure_analysis()
print("\n===== 自动识别结果 =====")
print(f"order         : {r['order']}")
print(f"|m|           : {r['magnetization']:.4f}")
print(f"q* (分数坐标) : {r['q_stars']}")
print(f"拓扑荷 Q      : {r['top_charge']:.3f}")
print(f"磁单胞基矢    : {r['cell_matrix']}  (超胞格点坐标)")
print(f"超胞含单胞数  : {r['cell_repeats']}")
cs = r['cell_spins'][:, :, 0, 2]
print(f"单胞 mz 范围   : [{cs.min():.2f}, {cs.max():.2f}]")
