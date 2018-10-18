import tensorflow as tf
import numpy as np
# data container
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='W')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='b')

# load
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("W", sess.run(W))
    print("b", sess.run(b))
