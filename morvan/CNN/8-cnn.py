import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data
# one hot : a encoder method
# eg 5 classes [1,0,0,0,0]
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.0})
    return result


def weight_variable(shape):
    # random value frome normal distribution truncated(pianduan!)
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, W):
    # stride[0]/[3] is fixed
    # stride[1]/[2] : x/y axis stride
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_poo_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholder
xs = tf.placeholder(tf.float32, shape=[None, 784])
ys = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
# n,w,h,c
# n=1 meaning any number of images
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# net

# conv1
# w,h,c,n
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# output : n*28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# output : n*14*14*32
h_pool1 = max_poo_2x2(h_conv1)

# conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# n*14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# n*7*7*64
h_pool2 = max_poo_2x2(h_conv2)

# FC1
# reshape to a col vector
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# loss
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

# train
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# sess
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(
        train_step, feed_dict={
            xs: batch_xs,
            ys: batch_ys,
            keep_prob: 0.5
        })
    if (i % 50 == 0):
        print(
            compute_accuracy(mnist.test.images[:1000],
                             mnist.test.labels[:1000]))
