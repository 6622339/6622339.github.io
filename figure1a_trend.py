import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 模拟论文数据
years = np.arange(1980, 2024)
np.random.seed(42)
d_land = 100 + 0.3*(years-1980) + np.random.normal(0, 8, len(years))

# 计算趋势
slope, intercept, r_val, p_val, std_err = linregress(years, d_land)
print(f"✅ 趋势斜率: {slope:.2f} km/年 (p值: {p_val:.3f})")

# 绘图
plt.figure(figsize=(10,5))
plt.plot(years, d_land, 'o-')
plt.plot(years, intercept + slope*years, 'r--')
plt.xlabel('Year')
plt.ylabel('D_land (km)')
plt.title('Tropical Cyclone Rainfall Extends Inland')
plt.savefig('figure1a.png', dpi=300)
plt.show()