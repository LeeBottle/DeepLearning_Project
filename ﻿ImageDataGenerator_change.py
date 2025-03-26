import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
batch_siz=128

generator_1=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
hist_1=cnn.fit(generator_1.flow(x_train,y_train,batch_size=batch_siz),epochs=10,validation_data=(x_test,y_test),verbose=2)
generator_2=ImageDataGenerator(featurewise_center=True,samplewise_center=True,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
hist_2=cnn.fit(generator_2.flow(x_train,y_train,batch_size=batch_siz),epochs=10,validation_data=(x_test,y_test),verbose=2)
generator_3=ImageDataGenerator(featurewise_std_normalization=True,samplewise_std_normalization=True,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=True)
hist_3=cnn.fit(generator_3.flow(x_train,y_train,batch_size=batch_siz),epochs=10,validation_data=(x_test,y_test),verbose=2)
generator_4=ImageDataGenerator(zca_whitening=True,zca_epsilon=1e10,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=True)
hist_4=cnn.fit(generator_4.flow(x_train,y_train,batch_size=batch_siz),epochs=10,validation_data=(x_test,y_test),verbose=2)
res=cnn.evaluate(x_test,y_test,verbose=0)

print("정확률은",res[1]*100)


import matplotlib.pyplot as plt

plt.plot(hist_1.history['accuracy'])
plt.plot(hist_1.history['val_accuracy'])
plt.title('API_1 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])
plt.title('API_1 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('API_2 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('API_2 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('API_3 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
lt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('API_3 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_4.history['accuracy'])
plt.plot(hist_4.history['val_accuracy'])
plt.title('API_4 Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

plt.plot(hist_4.history['loss'])
plt.plot(hist_4.history['val_loss'])
plt.title('API_4 Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()