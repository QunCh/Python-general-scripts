from scipy.stats import binom
import numpy as np

n, p = 15, 0.5
alist = []
for i in range(16):
    
    alist.append(binom.pmf(i,n,p))
sum(alist[:6])*2
alist[-1]

x = np.arange(binom.ppf(0.01, n, p),binom.ppf(0.99, n, p))