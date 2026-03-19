import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind

# ======================
# 1. 模拟WRF模式数据（论文两种情景：当前SST vs 升温SST）
# ======================
np.random.seed(42)
# 模拟西北太平洋区域经纬度（台风核心影响区）
lon = np.linspace(100, 160, 60)
lat = np.linspace(0, 30, 30)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 情景1：当前SST（CTL）- 基准降雨分布
rain_ctl = np.random.rand(30, 60) * 20
# 增强沿海降雨
rain_ctl[(lat_grid > 10) & (lat_grid < 25) & (lon_grid > 110) & (lon_grid < 150)] *= 2

# 情景2：SST升温1℃（WARM）- 降雨向内陆延伸
rain_warm = rain_ctl.copy()
# 内陆区域降雨增加（模拟向内陆延伸）
rain_warm[(lat_grid > 10) & (lat_grid < 25) & (lon_grid > 120) & (lon_grid < 140)] *= 1.8
# 计算两种情景的降雨差异
rain_diff = rain_warm - rain_ctl

# 统计检验（论文方法：两组情景均值差异）
ctl_mean = np.mean(rain_ctl[(lat_grid > 10) & (lat_grid < 25)])
warm_mean = np.mean(rain_warm[(lat_grid > 10) & (lat_grid < 25)])
t_stat, p_val = ttest_ind(rain_ctl.flatten(), rain_warm.flatten())

print(f"✅ 当前SST情景降雨均值：{ctl_mean:.1f} mm/day")
print(f"✅ 升温SST情景降雨均值：{warm_mean:.1f} mm/day")
print(f"✅ 两组情景差异显著性：t={t_stat:.2f}, p={p_val:.3f}")

# ======================
# 2. 绘制论文图3（三面板WRF模拟结果）
# ======================
fig = plt.figure(figsize=(18, 6))

# 面板1：当前SST情景（CTL）
ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax1.set_extent([100, 160, 0, 30], crs=ccrs.PlateCarree())  # 聚焦西北太平洋
cont1 = ax1.contourf(lon, lat, rain_ctl, cmap='Blues', levels=20, vmin=0, vmax=40)
ax1.set_title('CTL (Current SST)', fontsize=12)
plt.colorbar(cont1, ax=ax1, shrink=0.8, pad=0.05)

# 面板2：升温SST情景（WARM）
ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax2.set_extent([100, 160, 0, 30], crs=ccrs.PlateCarree())
cont2 = ax2.contourf(lon, lat, rain_warm, cmap='Blues', levels=20, vmin=0, vmax=40)
ax2.set_title('WARM (SST +1℃)', fontsize=12)
plt.colorbar(cont2, ax=ax2, shrink=0.8, pad=0.05)

# 面板3：情景差异（WARM - CTL）
ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax3.set_extent([100, 160, 0, 30], crs=ccrs.PlateCarree())
cont3 = ax3.contourf(lon, lat, rain_diff, cmap='RdBu_r', levels=15, vmin=-10, vmax=10)
ax3.set_title('Difference (WARM - CTL)', fontsize=12)
plt.colorbar(cont3, ax=ax3, shrink=0.8, pad=0.05)

# 整体标题
fig.suptitle('WRF Simulation: TC Rainfall Under Different SST Scenarios', fontsize=14, y=1.02)
plt.tight_layout()

# 保存高清图片
plt.savefig('figure3_wrf_simulation.png', dpi=300, bbox_inches='tight')
plt.show()