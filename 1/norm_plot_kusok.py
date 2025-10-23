#пример графической визуализации

import pandas as pd

#8 Нормализация данных

import numpy as np
from sklearn import preprocessing
df18=df3[['budget',"imdb_score"]]
df19=df18.dropna()
scaler_norm = preprocessing.MinMaxScaler()
scaler_range = preprocessing.MinMaxScaler()
x=scaler_range.fit_transform(df19[["budget"]])
df19 = df19.copy()
df19["budget"] = x[0:]
x=scaler_norm.fit_transform(df19[["imdb_score"]])
df19["imdb_score"] = x[0:]
print(df19)

#scaler_norm = preprocessing.StandardScaler
#Normalizer


# In[9]:


import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from numpy import arange
       


# In[10]:


plt.figure(figsize=(5, 5))
histData = plt.hist(df19.budget, bins = 500)
range_ = arange(-0.1, 0.1, 0.0005)
coefY = 0.5*len(df19.budget) * (histData[1][1]- histData[1][0])
plt.plot(range_,
         [norm(df19.budget.mean(), df19.budget.std()).pdf(x) * coefY for x in range_], #probability density function
         color='r')
plt.grid()
plt.show()


# In[11]:


plt.figure(figsize=(10, 6))
histData = plt.hist(df19.imdb_score)
range_ = arange(0, 1, 0.05)
coefY = len(df19.imdb_score) * (histData[1][1] - histData[1][0])
plt.plot(range_,
         [norm(df19.imdb_score.mean(), df19.imdb_score.std()).pdf(x) * coefY for x in range_], #probability density function
         color='r')
plt.grid()

plt.show()


# In[12]:


from scipy import stats
reg=stats.probplot(list(df19["budget"]),plot=plt,fit=True)
plt.show()


# In[13]:


from scipy import stats
reg=stats.probplot(list(df19["imdb_score"]),plot=plt,fit=True)
plt.show()


# In[14]:


df30=df3.assign(budget=(df3["budget"]-min(df3["budget"]))/(max(df3["budget"])-min(df3["budget"])))
hyst=df30["budget"]
hyst.hist()
