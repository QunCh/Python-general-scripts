from scipy import stats
import numpy as np
import pandas as pd

# Normal Distribution
s = np.random.normal(5,5, 1000)
df = pd.DataFrame(s)
df.to_excel('simulator.xlsx', index = False)