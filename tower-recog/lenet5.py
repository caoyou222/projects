import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt 
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

keep_prob=tf.placeholder("float")
learning_rate = 0.001

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [4096])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=n_out),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=n_out),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x,W,b):
    h_conv = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
    conv = tf.nn.tanh(tf.nn.bias_add(h_conv,b))
    return tf.layers.batch_normalization(conv, training=True)

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

# convolution layer
def lenet5_layer(layer,weight,bias,phase_train):
    W_conv=weight_variable(weight)
    b_conv=bias_variable(bias)
    h_conv=conv2d(layer,W_conv,b_conv)
    conv_bn = batch_norm(h_conv,bias,phase_train)
    return max_pool_2x2(conv_bn)

# connected layer
def dense_layer(layer,weight,bias):
    b_fc=bias_variable(bias)
    return tf.nn.tanh(tf.matmul(layer,weight)+b_fc)

def main():
    time_start = time.time()
    image, label = read_and_decode("tower.tfrecords")
    img_train_batch, labels_train_batch = tf.train.shuffle_batch([image, label],batch_size=20,capacity=15000,min_after_dequeue=6000,num_threads=2)
    labels_train_batch = tf.one_hot(labels_train_batch,depth=3) #resize the label into an array
    label = tf.one_hot(label, depth=3)
    sess=tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    regularizer = tf.contrib.layers.l2_regularizer(0.001)

    # input layer
    x=tf.placeholder("float",shape=[None,4096])
    y_=tf.placeholder("float",shape=[None,3])
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # first layer
    x_image=tf.reshape(x,[-1,64,64,1])
    layer=lenet5_layer(x_image,[5,5,1,6],[6],phase_train)

    # second layer
    layer=lenet5_layer(layer,[5,5,6,16],[16],phase_train)

    # third layer
    W_conv3=weight_variable([5,5,16,120])
    b_conv3=bias_variable([120])
    if regularizer != None:
            tf.add_to_collection('losses',regularizer(W_conv3))
    layer=conv2d(layer,W_conv3,b_conv3)
    layer = tf.reshape(layer,[-1,9*9*120])

    # all connected layer
    w_fc1 = weight_variable([9*9*120,84])
    if regularizer != None:
            tf.add_to_collection('losses',regularizer(w_fc1))
    con_layer=dense_layer(layer,w_fc1,[84])

    # output
    w_fc2 = weight_variable([84,3])
    if regularizer != None:
            tf.add_to_collection('losses',regularizer(w_fc2))
    con_layer=dense_layer(con_layer,w_fc2,[3])
    y_conv=tf.nn.softmax(tf.nn.dropout(con_layer,0.5))

    # train and evalute
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    try:
        accr = []
        costs = []
        for i in range(2000):
            img_xs, label_xs = sess.run([img_train_batch, labels_train_batch])
            if i%100==0:
                train_accuracy=accuracy.eval(feed_dict={x:img_xs,y_:label_xs,keep_prob:1.0,phase_train: False})
                batch_cost = cross_entropy.eval(feed_dict={x:img_xs,y_:label_xs,keep_prob:1.0,phase_train: False})
                accr.append(train_accuracy)
                costs.append(batch_cost)
                print("step %d,training accuracy %g"%(i,train_accuracy))
                print("step %d,cost %g"%(i,batch_cost))
            train_step.run(feed_dict={x:img_xs,y_:label_xs,keep_prob:0.5,phase_train: True})
    except Exception as e:
        coord.request_stop(e)
    coord.request_stop()
    coord.join(threads)
    saver.save(sess,"./model.ckpt")
    sess.close()

    time_stop = time.time()
    print("Training Time: " + str(time_stop-time_start) + "s")

    # plt.plot(accr)
    # plt.ylabel('accuracy')
    # plt.xlabel('iterations (per 100)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    # plt.plot(costs)
    # plt.ylabel('costs')
    # plt.xlabel('iterations (per 100)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

if __name__=='__main__':
    main()
