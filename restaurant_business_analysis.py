#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("restaurant data.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


data = data.drop(['url', 'address', 'phone', 'dish_liked', 'menu_item', 'reviews_list'], axis = 1)
data.head()


# In[9]:


data.info()


# In[10]:


data.drop_duplicates(inplace = True)
data.shape


# In[11]:


data["rate"].unique()


# In[12]:


def handlerate (value):
    if(value == 'NEW' or value == '-'):
        return np.nan
    else:
        value = str(value).split('/')
        value = value[0]
        return float(value)
data['rate'] = data['rate'].apply(handlerate)
data['rate'].head()


# In[13]:


data.rate.isnull().sum()


# In[14]:


data.rate.fillna(data['rate'].mean(), inplace = True)
data['rate'].isnull().sum()


# In[15]:


data.isnull().sum()


# In[16]:


data.dropna(inplace = True)
data.head()


# In[17]:


data.isnull().sum()


# In[18]:


plt.figure(figsize = (10,5))
ax = sns.countplot(data['location'])
#plt.xticks(rotation = 90)


# In[19]:


plt.figure(figsize = (6,6))
sns.countplot(data['online_order'], palette = 'inferno')


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


# Split the data into features and target
X = data[['votes', 'online_order']]
y = data['rate']


# In[22]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


from sklearn.preprocessing import OneHotEncoder


# In[26]:


# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[27]:


# Fit and transform the categorical features in the training set
X_train_encoded = encoder.fit_transform(X_train)


# In[28]:


# Transform the categorical features in the testing set
X_test_encoded = encoder.transform(X_test)


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


# Initialize the Linear Regression model
model = LinearRegression()


# In[31]:


# Train the model
model.fit(X_train_encoded, y_train)


# In[32]:


print("Thank_You")

