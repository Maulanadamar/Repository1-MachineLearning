#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# In[16]:


iris_data = pd.read_csv("Iris.csv")


# In[17]:


X = iris_data.drop('Species', axis=1)  
y = iris_data['Species']                      


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


svm_model = SVC(kernel='linear', random_state=42)


# In[31]:


svm_model.fit(X_train, y_train)


# In[32]:


y_pred = svm_model.predict(X_test)


# In[35]:


accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)


# In[38]:


print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))


# In[ ]:




