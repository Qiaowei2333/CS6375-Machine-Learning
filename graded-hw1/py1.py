import tensorflow as tf

# define the computational graph
node1 = tf.constant(3.0)
# end of the computational graph

# run the computational graph
sess = tf.Session()
print(sess.run(node1))
