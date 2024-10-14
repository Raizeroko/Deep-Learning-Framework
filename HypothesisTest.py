from scipy import stats
import numpy as np



# 样本数据
method1 = np.array([0.75, 0.80, 0.78, 0.82, 0.76])
method2 = np.array([0.65, 0.68, 0.72, 0.70, 0.69])

# 独立样本t检验
t_stat, p_value = stats.ttest_ind(method1, method2)

print(f"T-statistic: {t_stat}, P-value: {p_value}")
