import tensorflow as tf

# define the computational graph
W = tf.Variable(10.0) 
b = tf.Variable(-1.0)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print( sess.run(linear_model, {x:[1,2,3,4]}) )

