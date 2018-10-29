import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

keep_prob=tf.placeholder("float")

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

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

# convolution layer
def lenet5_layer(layer,weight,bias):
    W_conv=weight_variable(weight)
    b_conv=bias_variable(bias)

    h_conv=conv2d(layer,W_conv)+b_conv
    return max_pool_2x2(h_conv)

# connected layer
def dense_layer(layer,weight,bias):
    W_fc=weight_variable(weight)
    b_fc=bias_variable(bias)

    return tf.matmul(layer,W_fc)+b_fc

def main():
    image, label = read_and_decode("tower.tfrecords")
    img_train_batch, labels_train_batch = tf.train.shuffle_batch([image, label],batch_size=50,capacity=15000,min_after_dequeue=6000,num_threads=2)
    labels_train_batch = tf.one_hot(labels_train_batch,depth=3) #resize the label into an array
    sess=tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("begin session")
    # input layer
    x=tf.placeholder("float",shape=[None,4096])
    y_=tf.placeholder("float",shape=[None,3])

    # first layer
    x_image=tf.reshape(x,[-1,64,64,1])
    layer=lenet5_layer(x_image,[5,5,1,6],[6])
    print("1st layer")
    # second layer
    layer=lenet5_layer(layer,[5,5,6,16],[16])
    print("2nd layer")
    # third layer
    W_conv3=weight_variable([5,5,16,120])
    b_conv3=bias_variable([120])
    print("3rd layer")
    layer=conv2d(layer,W_conv3)+b_conv3
    layer = tf.reshape(layer,[-1,9*9*120])


    # all connected layer
    con_layer=dense_layer(layer,[9*9*120,84],[84])
    print("connected layer")
    # output
    con_layer=dense_layer(con_layer,[84,3],[3])
    y_conv=tf.nn.softmax(tf.nn.dropout(con_layer,keep_prob))
    print(y_conv.shape)
    print("output layer")
    # train and evalute
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print("train")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    print("loop")
    try:
        for i in range(20000):
            img_xs, label_xs = sess.run([img_train_batch, labels_train_batch])
            if i%100==0:
                train_accuracy=accuracy.eval(feed_dict={
                    x:img_xs,y_:label_xs,keep_prob:1.0
                })
                print("step %d,training accuracy %g"%(i,train_accuracy))
            train_step.run(feed_dict={x:img_xs,y_:label_xs,keep_prob:0.5})
    except Exception, e:
        coord.request_stop(e)
    coord.request_stop()
    coord.join(threads)
    saver.save(sess,"./model.ckpt")
    sess.close()

    #print("Test accuracy %g"%accuracy.eval(feed_dict={
        #x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
    #}))

if __name__=='__main__':
    main()
