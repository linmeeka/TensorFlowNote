import tensorflow as tf
import numpy as np

# data
W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='W')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='b')

init = tf.global_variables_initializer()

save
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path:", save_path)
