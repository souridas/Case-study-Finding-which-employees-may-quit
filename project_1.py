# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:36:10 2021

@author: Souridas A
"""
import numpy as np
import pandas as pd
hr=pd.read_csv('hr_data.csv')
emp_sat=pd.read_excel('employee_satisfaction_evaluation.xlsx')
main_data=hr.set_index('employee_id').join(emp_sat.set_index('EMPLOYEE #'))
main_data=main_data.reset_index()
summary=main_data.describe()
main_data.fillna(main_data.mean(),inplace=True)
main_data=main_data.drop(columns='employee_id')
dept_wise=main_data.groupby('department').sum()
#Displaying Correlation matrix
import matplotlib.pyplot as plt
def corr(df,size=10):
    corr=df.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation='vertical')
    plt.yticks(range(len(corr.columns)),corr.columns)
#corr(main_data)    
#Encoding
categorical=['department','salary']
main_data=pd.get_dummies(main_data,columns=categorical,drop_first=True)
#test-train split
from sklearn.model_selection import train_test_split
x=main_data.drop(columns='left').values
y=main_data['left'].values
y=y.reshape((-1,1))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#normalising
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#logistic_regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
'''model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(accuracy_score(prediction,y_test))
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))'''
#Random-forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(accuracy_score(prediction,y_test))
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))
#DL
'''import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(9,activation='relu',input_dim=18))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=10,epochs=16,validation_data=(x_test,y_test))
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])'''
# Random forest gave the benchmark for the prediction accuracy
#Checking the features which high imporatnce in predicting outcome
feature_imp=pd.DataFrame(model.feature_importances_,index=pd.DataFrame(x_train).columns,columns=['importance']).sort_values('importance',ascending=False)