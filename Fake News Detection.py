#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# In[7]:


import pandas as py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[8]:


Data_Fake = py.read_csv('Fake.csv')
Data_True = py.read_csv('True.csv')


# In[9]:


Data_Fake.head()


# In[10]:


Data_True.head()


# In[11]:


Data_Fake['class'] = 0
Data_True['class'] = 1


# In[12]:


Data_Fake.shape, Data_True.shape


# In[13]:


Data_Fake_Manual_Testing = Data_Fake.tail(10)
for i in range(23480, 23470, -1):
    Data_Fake.drop([i], axis = 0, inplace = True)

Data_True_Manual_Testing = Data_True.tail(10)
for i in range(21416, 21406, -1):
    Data_True.drop([i], axis = 0, inplace = True)    


# In[14]:


Data_Fake.shape, Data_True.shape


# In[15]:


Data_Fake_Manual_Testing['class'] = 0
Data_True_Manual_Testing['class'] = 1


# In[16]:


Data_Fake_Manual_Testing.head(10)


# In[17]:


Data_True_Manual_Testing.head(10)


# In[18]:


data_merge = py.concat([Data_Fake, Data_True], axis = 0)
data_merge.head(10)


# In[19]:


data_merge.columns


# In[20]:


data = data_merge.drop(['title', 'subject', 'date'], axis = 1)


# In[21]:


data.isnull().sum()


# In[22]:


data = data.sample(frac = 1)


# In[23]:


data.head()


# In[24]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[25]:


data.columns


# In[26]:


data.head()


# In[27]:


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Raw string to avoid escape issues
    text = re.sub(r"\W", " ", text)  # Raw string for \W
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Raw string for \S
    text = re.sub(r'<.*?>+', '', text)  # Raw string for <>
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Handles punctuation
    text = re.sub(r'\n', '', text)  # Raw string for newline
    text = re.sub(r'\w*\d\w*', '', text)  # Raw string for word-number patterns
    return text


# In[28]:


data['text'] = data['text'].apply(wordopt)


# In[29]:


x = data['text']
y = data['class']


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[32]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[33]:


pred_lr = LR.predict(xv_test)


# In[34]:


LR.score(xv_test, y_test)


# In[35]:


print(classification_report(y_test, pred_lr))


# In[36]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[37]:


pred_dt = DT.predict(xv_test)


# In[38]:


DT.score(xv_test, y_test)


# In[39]:


pred_dt = DT.predict(xv_test)
print(classification_report(y_test, pred_dt))


# In[40]:


from sklearn.ensemble import GradientBoostingClassifier


# In[41]:


GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[42]:


pred_gb = GB.predict(xv_test)


# In[43]:


GB.score(xv_test, y_test)


# In[44]:


print(classification_report(y_test, pred_gb))


# In[45]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


RF = RandomForestClassifier(random_state=0)


# In[47]:


RF.fit(xv_train, y_train)


# In[48]:


pred_rf = RF.predict(xv_test)


# In[49]:


RF.score(xv_test, y_test)


# In[50]:


print(classification_report(y_test, pred_rf))


# In[51]:


def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = py.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(
    output_lable(pred_LR[0]),
    output_lable(pred_DT[0]),  # Ensure DT is included
    output_lable(pred_GB[0]),                
    output_lable(pred_RF[0])
))


# In[56]:


news = str(input())
manual_testing(news)


# In[ ]:




