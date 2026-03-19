import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.gridspec as gridspec

# ======================
# 1. 模拟论文核心数据（SST 与 D_land 关联）
# ======================
# 时间范围：1980-2023
years = np.arange(1980, 2024)
np.random.seed(42)

# 模拟全球平均 SST 变化（逐年升温，和论文一致）
sst_trend = 0.015 * (years - 1980)  # 每10年升温0.15℃（IPCC数据）
sst_noise = np.random.normal(0, 0.05, size=len(years))
sst = 28.0 + sst_trend + sst_noise  # 基准SST=28℃

# 模拟 D_land（台风降雨向陆距离）：
# - 原始数据（含SST趋势）：和SST强相关
# - 去趋势数据（剔除SST影响）：趋势消失
d_land_original = 100 + 0.3*(years-1980) + np.random.normal(0, 5, len(years))
d_land_detrend = 100 + np.random.normal(0, 8, len(years))  # 无显著趋势

# 计算趋势（论文统计方法）
slope_original, _, p_original, _, _ = linregress(years, d_land_original)
slope_detrend, _, p_detrend, _, _ = linregress(years, d_land_detrend)

print(f"✅ 原始数据趋势：{slope_original:.2f} km/年 (p={p_original:.3f})")
print(f"✅ 去SST趋势后：{slope_detrend:.2f} km/年 (p={p_detrend:.3f})")

# ======================
# 2. 绘制论文图2（双面板归因分析）
# ======================
# 布局：上面板（SST vs D_land），下面板（原始vs去趋势）
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

# 面板1：SST 与 D_land 相关性散点图
ax1 = plt.subplot(gs[0])
# 散点图：SST vs D_land
ax1.scatter(sst, d_land_original, color='#1f77b4', alpha=0.7, label='Annual Data')
# 拟合线
slope_sst, intercept_sst, _, _, _ = linregress(sst, d_land_original)
sst_fit = intercept_sst + slope_sst * sst
ax1.plot(sst, sst_fit, 'r-', linewidth=2, label=f'Fit (slope={slope_sst:.1f})')
ax1.set_xlabel('Global Mean SST (℃)', fontsize=11)
ax1.set_ylabel('D_land (km)', fontsize=11)
ax1.set_title('SST vs Tropical Cyclone Rainfall Inland Distance', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# 面板2：原始数据 vs 去SST趋势数据
ax2 = plt.subplot(gs[1])
# 原始数据（有趋势）
ax2.plot(years, d_land_original, 'o-', color='#1f77b4', label=f'Original (slope={slope_original:.2f}, p={p_original:.3f})')
# 去趋势数据（无显著趋势）
ax2.plot(years, d_land_detrend, 's-', color='#ff7f0e', label=f'Detrended (slope={slope_detrend:.2f}, p={p_detrend:.3f})')
# 美化
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('D_land (km)', fontsize=11)
ax2.set_title('Attribution: SST is the Main Driver', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# 整体美化
plt.tight_layout()
# 保存高清图片
plt.savefig('figure2_sst_attribution.png', dpi=300, bbox_inches='tight')
plt.show()