import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#x有5個特徵直
dim=5
#y有1個答案
category=1
#有100筆資料
num=100
#產生資料
a=np.linspace(0,1,num*dim)
x=np.reshape(a,(num,dim))
y=np.array(x.sum(axis=1))
#分割資料
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,activation=tf.nn.relu,input_dim=dim))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))

model.compile(optimizer='sgd',loss='mse',metrics=['mse','mape','mae',tf.compat.v1.keras.losses.cosine_proximity])
history=model.fit(x_train,y_train,epochs=4000,batch_size=len(y_test))
score=model.evaluate(x_test,y_test,batch_size=128)
print("test cost:{}".format(score))
y_predict=model.predict(x_test)
x2=x[:,2]
print(x2[:5])
x_test2=x_test[:,2]


plt.plot(history.history['mse'])
plt.plot(history.history['mape'])
plt.plot(history.history['mae'])
plt.plot(history.history['cosine_similarity'])
plt.legend(['mse','mae','mape','cos'], loc='upper right')
plt.show()






