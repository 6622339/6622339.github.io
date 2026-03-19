import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ======================
# 1. 模拟全球台风降雨频率数据（和论文分布一致）
# ======================
# 生成全球经纬度网格
lon = np.linspace(-180, 180, 360)  # 经度
lat = np.linspace(-90, 90, 180)    # 纬度
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 模拟台风活跃区的降雨频率（和论文一致：西北太平洋>北大西洋>南半球）
np.random.seed(42)
rain_freq = np.random.rand(180, 360) * 10  # 基础噪声

# 增强西北太平洋（台风最活跃区）
rain_freq[(lat_grid > 0) & (lat_grid < 30) & (lon_grid > 100) & (lon_grid < 180)] *= 5
# 增强北大西洋
rain_freq[(lat_grid > 0) & (lat_grid < 30) & (lon_grid > -90) & (lon_grid < -20)] *= 3
# 增强南半球澳洲附近
rain_freq[(lat_grid < 0) & (lat_grid > -30) & (lon_grid > 110) & (lon_grid < 180)] *= 2

# ======================
# 2. 绘制论文图1c（全球空间分布图）
# ======================
fig = plt.figure(figsize=(14, 7))
# 使用 cartopy 绘制地图投影
ax = plt.axes(projection=ccrs.PlateCarree())

# 添加地图要素（海岸线、陆地、海洋）
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.LAND, color='#f0f0f0')
ax.add_feature(cfeature.OCEAN, color='#e6f3ff')
ax.set_global()  # 显示全球

# 绘制降雨频率填色图（和论文配色一致）
contour = ax.contourf(lon, lat, rain_freq, 
                      transform=ccrs.PlateCarree(),
                      cmap='Blues',  # 论文同款蓝色系
                      levels=15,     # 颜色分级
                      vmin=0, vmax=50)

# 添加色标（图例）
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('TC Rainfall Frequency (%)', fontsize=12)

# 美化标题和样式
plt.title('Global Distribution of Tropical Cyclone Rainfall (1980-2023)', fontsize=14, pad=20)
plt.tight_layout()

# 保存高清图片（论文级分辨率）
plt.savefig('figure1c_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()