# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:40:21 2020

@author: Prateek
"""


import numpy as np
import pandas as pd
from matplotlib import *
import pylab as plt
import seaborn as sns


df = pd.read_csv('Classified Data',index_col=0)

df.head()

#we see the dataset is perfectly balanced
df.info()

df.isnull().sum()

list(df)

#standardize the variable
from sklearn.preprocessing  import StandardScaler
scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


sns.pairplot(df)

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'], test_size=0.30, 
                                                    random_state=101)


#now apply different algorithm KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

#Validate the model
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))


#choosing k value by error rate or accuracy rate

#error rate
error_rate = []

#will take some time
for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i= knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
        
        
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker= 'o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. Kvalue')
plt.xlabel('k')
plt.ylabel('Errror Rate')




#by accuracy rate
accuracy_rate = []

#will take some time
from sklearn.model_selection import cross_val_score
for i in range(1,40):
        
        knn = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
        accuracy_rate.append(score.mean())
        

plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker= 'o',markerfacecolor='red',markersize=10)
plt.title('accuracy vs. Kvalue')
plt.xlabel('k')
plt.ylabel('accuracy')