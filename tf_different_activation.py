
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
iris=datasets.load_iris()
category=3
dim=4
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train,num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test,num_classes=(category))

def model(opt1):
  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=20,activation=tf.nn.relu,input_dim=dim))
  model.add(tf.keras.layers.Dense(units=20,activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
  model.compile(
      optimizer=opt1,
      loss=tf.keras.losses.categorical_crossentropy,
      metrics=['accuracy'])
  history=model.fit(x_train,y_train2,epochs=80,batch_size=16)
  return history
learning_rate=0.01
history_adam=model(tf.keras.optimizers.Adam(lr=learning_rate))
history_sgd=model(tf.keras.optimizers.SGD(lr=learning_rate))
history_rmsprop=model(tf.keras.optimizers.RMSprop(lr=learning_rate))
history_adagrad=model(tf.keras.optimizers.Adagrad(lr=learning_rate))
history_Adadelta=model(tf.keras.optimizers.Adadelta(lr=learning_rate))
history_nadam=model(tf.keras.optimizers.Nadam(lr=learning_rate))
history_mom=model(tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.9))
plt.plot(history_adam.history['accuracy'])
plt.plot(history_sgd.history['accuracy'])
plt.plot(history_rmsprop.history['accuracy'])
plt.plot(history_adagrad.history['accuracy'])
plt.plot(history_Adadelta.history['accuracy'])
plt.plot(history_nadam.history['accuracy'])
plt.plot(history_mom.history['accuracy'])

plt.title('optimizers acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['adam','sgd','rmsprop','adagrad','adadelta','nadam','mom'],
            loc='lower right')
plt.show()
