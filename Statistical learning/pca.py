import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
df = pd.read_csv(iris['filename'])

scaler = StandardScaler()
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
x_pca
pca.explained_variance_ratio_
pca.explained_variance_

fig = plt.figure(figsize = (6,5))
plt.scatter(x_pca[:,0],x_pca[:,1], c=Y, s=20)