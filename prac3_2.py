#!/usr/bin/env python
# coding: utf-8

# In[33]:


import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
sklearn.__version__


# In[5]:


#reading the csv dataset
df= pd.read_csv('/home/student/Downloads/diabetes.csv')
df


# In[3]:


# # #preprocessing
# zero_not_accepted= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# print(df)
# for column in zero_not_accepted:
#     df[column]= df[column].replace(0, np.NaN)
#     mean= float(df[column].mean())
#     df[column]= df[column].replace(np.NaN, mean)
    
# df


# In[6]:


#splitting columns into input and output

x= df.iloc[:, 1:8]
y= df.iloc[:, -1:]


# In[7]:


x


# In[8]:


y


# In[9]:


#splitting into training and testing inputs and outputs
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2)


# In[36]:


print(x_train)
y_train


# In[11]:


#Standardizing the input: fitting to normal distribution and transforming

sc= StandardScaler()
x_sc_train= sc.fit_transform(x_train)
x_sc_test= sc.fit_transform(x_test)


# In[37]:


#training the model

model= GaussianNB()
model.fit(x_sc_train, y_train)
model2 = LogisticRegression()
model2.fit(x_sc_train, y_train)


# In[28]:


#predicting output

y_pred= model.predict(x_sc_test)
y_pred


# In[29]:


#Evaluating the model

cm= confusion_matrix(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)
print("Correct Predictions: ", cm[0][0]+cm[1][1])
print(cm)
print("Accuracy: ", accuracy)


# In[38]:


y_pred2= model2.predict(x_sc_test)
y_pred2
cm= confusion_matrix(y_test, y_pred2)
accuracy= accuracy_score(y_test, y_pred2)
print("Correct Predictions: ", cm[0][0]+cm[1][1])
print(cm)
print("Accuracy: ", accuracy)

