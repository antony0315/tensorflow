from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
iris=datasets.load_iris()
category=3
dim=4
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train,num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test,num_classes=(category))
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=20,activation=tf.nn.relu,input_dim=dim))
model.add(tf.keras.layers.Dense(units=20,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train,y_train2,
          epochs=200,
          batch_size=16)
score=model.evaluate(x_test,y_test2,batch_size=128)
print("score=",score)
#測試
predict=model.predict(x_test)
print("predict:",predict)
print(np.argmax(predict,axis=1))
print(y_test[:])

from tensorflow.keras.callbacks import TensorBoard
tensorboard=TensorBoard(log_dir='logs')
history=model.fit(x_train,y_train2,epochs=200, batch_size=16,
                  callbacks=[tensorboard],
                  verbose=1)
#存取
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")



