import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#initialize bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#build convolutional layer: x is a 4d tensor, shape=[batch, height, width, channels]
#stride = 1, padding = 'SAME'(use every pixel), 'VALID' means disregard edges
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#build pooling layer
#x is a 4d tensor shape=[batch, height, width, channels]
#ksize is 2*2, stride is 2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")
def Classified(path):
#Placeholder
	x = tf.placeholder("float", shape=[None, 784]) #image
	y_ = tf.placeholder("float", shape=[None, 10]) #which digit

#initialize weight


#C1
	W_conv1 = weight_variable([5,5,1,6]) #5*5 filter, 1 channel, 6 feature maps

	b_conv1 = bias_variable([6]) #output size

	x_image = tf.reshape(x, [-1,28,28,1]) #reshape x into a 4d tensor shape=[batch, 28, 28, 1]

#di the convolution, then activate by ReLU, then max_pooling
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1) #output for the first layer shape=[batch, 14, 14, 6]

#C3
	W_conv2 = weight_variable([5,5,6,16])
	b_conv2 = weight_variable([16])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

#Fully connected layer with 1024 neurons
	W_fc1 = weight_variable([7*7*16, 120]) #size = pool2 size * #of neurons from S4
	b_fc1 = bias_variable([120])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Reduce overfitting, add dropout before output layer
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer with softmax
	W_fc2 = weight_variable([120, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	src = cv2.imread(path)
	src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	dst = cv2.resize(src, (28, 28), interpolation=cv2.INTER_CUBIC)

	picture = np.zeros((28, 28))
	for i in range(0, 28):
		for j in range(0, 28):
			picture[i][j] = (255 - dst[i][j])
	picture = picture.reshape([1, 784])

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, "/tmp/model.ckpt")
	print("Model restored")
	result = sess.run(y_conv, feed_dict={x:picture})


path = "sample.png"
print(Classified(path))