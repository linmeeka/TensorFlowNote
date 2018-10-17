import tensorflow as tf

# constant matrix.
# tensor:[a] a is a single element
# tensor:[a,b,c] a,b,c is a row which size is 1*3 =>[elements] =>col
# tensor:[[a],[b],[c]] a col which size is 3*1
# tensor:[[a,b,c],[d,c,e]] a matrix which size is 2*3 =>[col]=>matrix
# 1*2
matrix1 = tf.constant([[3, 3]])
# 2*1
matrix2 = tf.constant([[2], [2]])
# matrix multiply np.dot(m1,m2)
product = tf.matmul(matrix1, matrix2)

# method 1

sess = tf.Session()
# run product (m1*m2)
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
