import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt 
import time
import cv2
from PIL import Image, ImageFilter
from tensorflow.python.client import device_lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

keep_prob=tf.placeholder("float")
learning_rate = 0.0001
batch_size = 50
class_num = 3

def read_and_decode(filename, b_size):
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [1024])
    label = tf.cast(features['label'], tf.int32)

    img_train_batch, labels_train_batch = tf.train.shuffle_batch([img, label],batch_size=b_size,capacity=6000,min_after_dequeue=5000,num_threads=2)
    labels_batch = tf.one_hot(labels_train_batch,depth=class_num)

    return img_train_batch, labels_batch

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x,W,b):
    h_conv = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(h_conv,b))

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

# convolution layer
def lenet5_layer(layer,weight,bias):
    W_conv=weight_variable(weight)
    b_conv=bias_variable(bias)

    h_conv=conv2d(layer,W_conv,b_conv)
    return max_pool_2x2(h_conv)

# connected layer
def dense_layer(layer,weight,bias):
    W_fc=weight_variable(weight)
    b_fc=bias_variable(bias)

    return tf.matmul(layer,W_fc)+b_fc

def main():
    time_start = time.time()
    img_train_batch, labels_train_batch = read_and_decode('tower-train-32.tfrecords', batch_size)

    x=tf.placeholder("float",shape=[None,1024])
    y_=tf.placeholder("float",shape=[None,class_num])

    # first layer
    x_image=tf.reshape(x,[-1,32,32,1])
    layer=lenet5_layer(x_image,[5,5,1,6],[6])
    w1 = weight_variable([5,5,1,6])
    # second layer
    layer=lenet5_layer(layer,[5,5,6,16],[16])

    # third layer
    W_conv3=weight_variable([5,5,16,120])
    b_conv3=bias_variable([120])

    layer=conv2d(layer,W_conv3,b_conv3)
    layer=tf.reshape(layer,[-1,120])

    # all connected layer
    con_layer=dense_layer(layer,[120,84],[84])

    # output
    con_layer=dense_layer(con_layer,[84,class_num],[class_num])
    y_conv=tf.nn.softmax(tf.nn.dropout(con_layer,keep_prob))

    # train and evalute
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv))
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

    saver = tf.train.Saver()
    
    # Restore model part
    with tf.Session() as sess:
        path = "./fiver.png" #path for image
        src = cv2.imread(path)
        #cv2.imshow("fiver", src)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.resize(src, (32,32), interpolation=cv2.INTER_CUBIC)

        picture = dst.reshape(1, 1024)


        saver.restore(sess, './stable_32/tower_model_stable.ckpt') #path for the model
        print("Model restored")
    
        result = sess.run(y_conv, feed_dict={x:picture,keep_prob: 1.0})
        if np.argmax(result[0]) == 1:
            final_result = "This is turtle."
        elif np.argmax(result[0]) == 0:
            final_result = "This is box."
        else:
            final_result = "This is fiver."
        print(final_result)


if __name__=='__main__':
    main()
