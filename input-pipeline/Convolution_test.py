import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential


batch_size = 100
epoch = 12

img_x, img_y = 28, 28

(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5),
                 strides = (1, 1)
                 activation = 'relu'
                 input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(kernel_size = (5, 5),
                 strides = (1, 1)
                  activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2). strides = (2, 2)))
model.add(Flatten())
model.add(Dense())
model.add(Flatten())
model.add(Dense())
"""
