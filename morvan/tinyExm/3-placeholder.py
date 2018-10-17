import tensorflow as tf

# to place input data
# feed data during running
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

data = {input1: [7.], input2: [2.]}

with tf.Session() as sess:
    print(sess.run(output, data))
