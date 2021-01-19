from scipy.stats import binom
from scipy import stats
import numpy as np

# 根据t值计算p-value
stats.t.sf(2.17,df=4.73)*2

n, p = 15, 0.5
alist = []
for i in range(16):
    
    alist.append(binom.pmf(i,n,p))
sum(alist[:6])*2
alist[-1]

x = np.arange(binom.ppf(0.01, n, p),binom.ppf(0.99, n, p))