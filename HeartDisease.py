# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:08:11 2019

@author: yimin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
sns.set_style("darkgrid")


#load Data
dt = pd.read_csv('heart.csv')
#head 10
dt.head(10)
#check data
dt.info()

plt.figure(figsize=(10,8))
sns.heatmap(dt.corr(),annot=True,cmap='YlOrBr',fmt='.2f',linewidths=2)

#There's not such correlations between these variables

sns.distplot(dt['age'],color='Brown',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})

#The targets ages is around 40 to 60
#test age range
fig,ax=plt.subplots(figsize=(30,7))
plt.subplot(1, 3, 1)
age_bins = [20,30,40,50,60,70,80]
dt['bin_age']=pd.cut(dt['age'], bins=age_bins)
g1=sns.countplot(x='bin_age',data=dt ,hue='target',palette='Blues_d',linewidth=3)
g1.set_title("Age vs Heart Disease")

#test Cholestorals
plt.subplot(1, 3, 2)
cho_bins = [100,150,200,250,300,350,400,450]
dt['bin_chol']=pd.cut(dt['chol'], bins=cho_bins)
g2=sns.countplot(x='bin_chol',data=dt,hue='target',palette='Blues_d',linewidth=3)
g2.set_title("Cholestoral vs Heart Disease")
#200

#test Thal
plt.subplot(1, 3, 3)
thal_bins = [60,80,100,120,140,160,180,200,220]
dt['bin_thal']=pd.cut(dt['thalach'], bins=thal_bins)
g3=sns.countplot(x='bin_thal',data=dt,hue='target',palette='Blues_d',linewidth=3)
g3.set_title("Thal vs Heart Disease")
# 140-180

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
x1=sns.countplot(x='cp',data=dt,hue='target',palette='PuBu',linewidth=3)
x1.set_title('Chest pain type')
#Chest pain type 2 

plt.subplot(132)
x2=sns.countplot(x='thal',data=dt,hue='target',palette='PuBu',linewidth=3)
x2.set_title('Thal')
#People with thal 2 

plt.subplot(133)
x3=sns.countplot(x='slope',data=dt,hue='target',palette='PuBu',linewidth=3)
x3.set_title('slope of the peak exercise ST segment')
#Slope 2 

fig,ax=plt.subplots(figsize=(16,6))
plt.subplot(121)
s1=sns.boxenplot(x='sex',y='age',hue='target',data=dt,palette='PuBu',linewidth=3)
s1.set_title("Figure 1")
# most of females having heart disease range from 40-70yrs and men from 40-60yrs

plt.subplot(122)
s2=sns.pointplot(x='sex',y='age',hue='target',data=dt,palette='PuBu',capsize=.2)
s2.set_title("Figure 2")
# mean age for female with heart disease around 54yrs and for males around 51yrs

fig,ax=plt.subplots(figsize=(16,6))
sns.pointplot(x='age',y='cp',data=dt,color='Blue',hue='target')
plt.title('Age vs Cp')
#People with heart disease tend to have higher 'cp' at all ages except age 45 and 49

fig,ax=plt.subplots(figsize=(16,6))
sns.lineplot(y='thalach',x='age',data=dt,hue="target",style='target',palette='PuBu',markers=True, dashes=False,err_style="bars", ci=68)
plt.title('Age vs Thalach')
#Thalach always high in people having heart disease and as age increases the thalach seems to reduce and other factors might play a role in heart disease
sns.pointplot(x='sex',y='thal',data=dt,hue='target',markers=["o", "x"],capsize=.2,palette='BuGn')
#Both males and females without heart disease have higher thal value and males with heart diseases tend to have higher thal than females

sns.countplot(x='ca',data=dt,hue='target',palette='YlOrBr',linewidth=3)
# People with 'ca' as 0 have highest chance of heart disease

sns.countplot(x='ca',data=dt,hue='target',palette='YlOrBr',linewidth=3)
# People with 'ca' as 0 have highest chance of heart disease

sns.countplot(x='slope',hue='target',data=dt,palette='YlOrBr',linewidth=3)
#Slope 2 has highest people with heart disease

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
old_bins = [0,1,2,3,4,5,6]
dt['bin_old']=pd.cut(dt['oldpeak'], bins=old_bins)
sns.countplot(x='bin_old',hue='target',data=dt,palette='bwr',linewidth=3)
plt.title("Figure 1")
#Figure 1: As the value of oldpeak increases the rate of heart disease decreases

plt.subplot(132)
sns.boxplot(x='slope',y='oldpeak',data=dt,hue='target',palette='bwr',linewidth=3)
plt.title("Figure 2")
#Figure 2: slope-s and target = 1; for s=0 --> Median Oldpeak=~1.4; for s=1 --> Median Oldpeak=~0.7; for s=2 --> Median Oldpeak=~0

plt.subplot(133)
sns.pointplot(x='slope',y='oldpeak',data=dt,hue='target',palette='bwr')
plt.title("Figure 3")
#Figure 3: As the value of slope increases the oldpeak values decrease and heart disease people have lower oldpeak
#start to clean data
dt.head()
dt.drop(['bin_age','bin_chol','bin_thal','bin_old'],axis=1,inplace=True)
dt.head()

dt.dtypes
#Conversion to categorical variables
dt['sex']=dt['sex'].astype('category')
dt['cp']=dt['cp'].astype('category')
dt['fbs']=dt['fbs'].astype('category')
dt['restecg']=dt['restecg'].astype('category')
dt['exang']=dt['exang'].astype('category')
dt['slope']=dt['slope'].astype('category')
dt['ca']=dt['ca'].astype('category')
dt['thal']=dt['thal'].astype('category')
dt['target']=dt['target'].astype('category')
dt.dtypes

y=dt['target']
dt=pd.get_dummies(dt,drop_first=True)
dt.head()

X=dt.drop('target_1',axis=1)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gaussian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test,predictions))
    print(name,accuracy_score(y_test,predictions))
from sklearn.metrics import classification_report,confusion_matrix

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
log_pred=logmodel.predict(X_test)
print(confusion_matrix(y_test,log_pred))
print(classification_report(y_test,log_pred))
print(accuracy_score(y_test,log_pred))
#Hyperparameter tuning for Logistic Regression
from sklearn.model_selection import GridSearchCV
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
h_logmodel = GridSearchCV(logmodel, hyperparameters, cv=5, verbose=0)
best_logmodel=h_logmodel.fit(X,y)
print('Best Penalty:', best_logmodel.best_estimator_.get_params()['penalty'])
print('Best C:', best_logmodel.best_estimator_.get_params()['C'])

logmodel=LogisticRegression(penalty='l1',C=2.7825594022071245)
logmodel.fit(X_train,y_train)
h_log_pred=logmodel.predict(X_test)
print(confusion_matrix(y_test,h_log_pred))
print(classification_report(y_test,h_log_pred))
print(accuracy_score(y_test,h_log_pred))
