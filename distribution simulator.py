from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns

# Normal Distribution
s = np.random.normal(5,5, 1000)
df = pd.DataFrame(s)
df.to_excel('simulator.xlsx', index = False)

np.array(np.random.random()

parray = np.random.random(size = 1000000)
test = parray/(1-parray)
a = np.log(test)
sns.distplot(a)
sns.scatterplot(test, a)

np.log(0.5)