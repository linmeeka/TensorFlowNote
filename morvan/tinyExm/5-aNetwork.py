# error when using tensorboard to visualize
# google.protobuf.message.DecodeError: Error parsing message
# not get clear yet

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    # define layer name
    with tf.name_scope('layer'):
        # define weight name
        with tf.name_scope('W'):
            Weights = tf.Variable(
                tf.random_normal([in_size, out_size]), name='W')
        # define biases name
        with tf.name_scope('b'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        # define ...
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# data
# linspace : genera a dengchashulie!
# data size:300
# x:1 dimension
# y:1 dimension
# actually done is fit a curve
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# None: any sample number
# easy to implement mini-batch
# name will be showed on tensorbord 'input' layer
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# loss : the mean of all samples' diff
with tf.name_scope('loss'):
    # # 300*1
    # _sq = tf.square(prediction - ys)
    # # 300,reduction_indices=[1],sum through col.dimension reduction
    # _sum = tf.reduce_sum(_sq, reduction_indices=[1])
    # # cal mean through row,300=>1 ,get a singel number
    # loss = tf.reduce_mean(_sum)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


