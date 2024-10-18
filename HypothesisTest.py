from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# #session1 seed3
#corrected p-values: [0.00463867 0.01672363 0.07381185 0.76153564]
dan_acc = [80.88, 79.79, 65.97, 69.42, 75.66, 78.08, 66.03, 75.07, 66.03, 78.96, 71.33, 68.92, 68.53, 72.57, 72.92]
udda_acc = [58.99, 42.69, 70.65, 63.85, 69.30, 81.50, 64.05, 74.72, 86.71, 78.08, 69.03, 78.34, 70.18, 81.26, 100.00]
mda_acc = [91.46, 86.65, 63.35, 75.90, 92.63, 98.06, 81.14, 77.17, 77.20, 89.84, 87.86, 67.74, 82.29, 74.01, 87.98]
prpl_acc=[89.16, 100.00, 100.00 ,79.14 ,100.00 ,100.00 ,100.00 ,85.00 ,	95.85 ,93.55 ,	87.68 ,	93.52 ,	92.63 ,	93.13 ,	84.00 ]  # 空列表表示没有对应的数值
ours_acc =[98.06 ,	98.06 ,	58.04 ,	98.06 ,	92.22 ,	92.96 ,	98.06 ,	98.06 ,	98.06 ,	98.06 ,	61.64 ,	97.02 ,	98.06 ,	79.76 ,	80.17   ]

# 计算与ours的p值,
p_values = []
# for acc in [fbcsp_bci_acc, eeg_bci_acc, deep_bci_acc, fbc_bci_acc]:
#     _, p_value = wilcoxon(acc, dmsanet_bci_acc)
#     p_values.append(p_value)
for acc in [dan_acc,udda_acc,mda_acc,prpl_acc]:
    # 执行两侧的Wilcoxon符号秩检验
    _, p_value = wilcoxon(acc, ours_acc)
    p_values.append(p_value)
# 进行FDR检验
rejected, corrected_p_values = fdrcorrection(p_values)

print('p-values:', p_values)
print('corrected p-values:', corrected_p_values)
print('rejected:', rejected)