import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ======================
# 1. 模拟台风强度与 D_land 数据（论文结论：正相关）
# ======================
np.random.seed(42)
n_samples = 200  # 模拟200个台风样本

# 模拟台风强度（最大风速，单位：kt）
intensity = np.random.uniform(34, 140, size=n_samples)  # 热带风暴到超强台风
# 模拟向陆距离 D_land：强度越强，D_land 越大（正相关）
d_land = 80 + 0.6*intensity + np.random.normal(0, 15, size=n_samples)

# 线性回归分析
slope, intercept, r_val, p_val, std_err = linregress(intensity, d_land)
print(f"✅ 强度与 D_land 斜率: {slope:.2f} km/kt (p值: {p_val:.3f}, R²: {r_val**2:.2f})")

# ======================
# 2. 绘制散点图 + 拟合线（论文补充图风格）
# ======================
plt.figure(figsize=(9, 6))

# 散点图：台风强度 vs 向陆距离
plt.scatter(intensity, d_land, color='#2ca02c', alpha=0.6, label='TC Samples')

# 拟合线
fit_line = intercept + slope*intensity
plt.plot(intensity, fit_line, 'r-', linewidth=2, label=f'Fit (slope={slope:.2f}, p={p_val:.3f})')

# 美化
plt.xlabel('TC Maximum Wind Speed (kt)', fontsize=12)
plt.ylabel('Rainfall Inland Distance D_land (km)', fontsize=12)
plt.title('TC Intensity vs Rainfall Inland Extension', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('figure4_intensity_vs_dland.png', dpi=300, bbox_inches='tight')
plt.show()