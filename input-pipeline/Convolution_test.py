import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from Keras.layers import Conv2D, MaxPooling2D

batch_size = 130
num_classes = 10
epochs = 12

#inputting image dimensions

img_rows, img_columns = 28,28

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape(img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

##x_train /= 255
##x_test /= 255

model = Sequential()
