import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
import os
(x_train,y_train),(x_test,y_test)=boston_housing.load_data()

classes=['CRIW','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATTION','B','LSTAT']
print(classes)

data=pd.DataFrame(x_train,columns=classes)
data['MEDV']=pd.Series(data=y_train)
print(data.head())
print(data.describe())


data.to_csv('C:/Users/anton/OneDrive/桌面/learn_python/tensorflow/boston.csv',sep=',')

writer=pd.ExcelWriter('C:/Users/anton/OneDrive/桌面/learn_python/tensorflow/boston.xlsx', engine='xlsxwriter')
data.to_excel(writer,sheet_name='Sheet1')
writer.save()
writer.close()


import seaborn as sns 
# sns.pairplot(data[['MEDV','CRIW','AGE','DIS','TAX']],diag_kind="kde")
# plt.show()




g=sns.PairGrid(data[['MEDV','CRIW','AGE','DIS','TAX']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot,cmap="Blues_d",n_levels=6)
plt.show()

from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

learning_rate=0.0001
opt1=tf.keras.optimizers.Nadam(lr=learning_rate)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(320,activation='relu',input_shape=[x_train.shape[1]]))
model.add(tf.keras.layers.Dense(units=640,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse',optimizer=opt1,metrics=['mae'])
history=model.fit(x_train,y_train,epochs=10000,batch_size=len(y_train))

#try:
#    with open('bostonmodel.h5','r') as load_weights:
#        model.load


print('start testing')
cost=model.evaluate(x_test,y_test)
print("test cost:{}".format(cost))
y_pred=model.predict(x_test)

print(y_pred[:10])
print(y_test[:10])

print(history.history.keys())
plt.plot(history.history['mae'])
plt.title('boston house price')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.legend(['train mae'],loc='upper right')
plt.show()

#model.save_weights('C:/Users/user/Desktop/bostonmodel.h5')

