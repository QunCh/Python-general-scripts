# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:59:43 2015

@author: jml
"""

import pandas
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt
import os
from statsmodels.formula.api import ols

os.chdir(r'D:\Github\Python general scripts\dataset')

data = pandas.read_csv('gapminder.csv', low_memory=False)

#setting variables you will be working with to numeric
data['internetuserate'] = pandas.to_numeric(data['internetuserate'], errors = 'coerce')
data['urbanrate'] = pandas.to_numeric(data['urbanrate'], errors = 'coerce')
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors = 'coerce')

data['incomeperperson']=data['incomeperperson'].replace(' ', numpy.nan)
"""
scat1 = seaborn.regplot(x="urbanrate", y="internetuserate", fit_reg=True, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Urban Rate and Internet Use Rate')

scat2 = seaborn.regplot(x="incomeperperson", y="internetuserate", fit_reg=True, data=data)
plt.xlabel('Income per Person')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Income per Person and Internet Use Rate')
"""
data_clean=data.dropna()
data_clean.corr()
print ('association between urbanrate and internetuserate')
print (scipy.stats.pearsonr(data_clean['urbanrate'], data_clean['internetuserate']))

print ('association between incomeperperson and internetuserate')
print (scipy.stats.pearsonr(data_clean['incomeperperson'], data_clean['internetuserate']))

model = ols(formula = 'incomeperperson ~ internetuserate', data = data_clean)
model.fit().summary()