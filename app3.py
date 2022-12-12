#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Iris.csv.xls')


# In[3]:


#df


# In[4]:


#DROPING THE ID COLUMN


# In[5]:


# Dropping the Id column

df.drop('Id', axis = 1, inplace = True)


# In[6]:


# Renaming the target column into numbers to aid training of the model


#df['Species']= df['Species' ].map({'Iris-setosa' :0, 'Iris-versicolor':1, 'Iris-virginica':2})

df['Species']= df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

#df


# In[7]:


# splitting the data into the columns which need to be trained(X) and the target column(y)

X = df. iloc[:, :-1]
y = df.iloc[:,-1]


# In[8]:


# splitting data into training and testing data with 30 % of data as testing data respectively


from sklearn.model_selection import train_test_split  #scikit learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[9]:


# importing the random forest classifier model and training it on the dataset
from sklearn. ensemble import RandomForestClassifier

# to use more then one algo then we use esemble technique 
#

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# In[10]:


# predicting on the test dataset

y_pred = classifier.predict (X_test)


# In[11]:


# finding out the accuracy

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)


# In[12]:


print(score)


# # Model Deployment
# 
# 
# # Using Streamlit library
# 
# # Streamlit is an open source app framework in Python language.
# 
# # It helps us create web apps for data science and machine learning in a short time. 
# 
# # Pickle in Python is primarily used in serializing and deserializing a Python object structure.
# 
# # It is the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.
# 

# In[13]:


#pickling the model

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[14]:


import pickle as pkl  
import streamlit as st  
from PIL import Image as img


# In[15]:


pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


# In[16]:


def welcome():
    return 'welcome all'
  


def prediction(sepal_length, sepal_width, petal_length, petal_width):  
   
    prediction = classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction
      
  
# main function in which we define our webpage 

def main():
      # giving the webpage a title
    st.title("Iris Flower Prediction")
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML Application </h1>
    </div>
    """
  
    
    st.markdown(html_temp, unsafe_allow_html = True)
      
   
    
    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")
    result =""

    if st.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:







#streamlit run app2.py

