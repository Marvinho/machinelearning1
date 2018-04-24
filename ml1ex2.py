from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 0, 3
data1 = np.random.normal(loc=mu,scale=sigma,size=50)
print(data1)
plt.xlabel("x afsdfjsf")
plt.ylabel("y afsdfjsf")
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)))
#cov1 = np.cov(data1)
#print(cov1)

mu, sigma = 0, 1
data2 = np.random.normal(loc=mu,scale=sigma,size=50)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)))
cov2 = np.cov(data1, data2)

print(cov2)
print(np.mean(data1))
print(np.mean(data2))
print(np.shape(data1))