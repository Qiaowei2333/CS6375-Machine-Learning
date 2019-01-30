# Experiments with deep learning neural nets

from our_mnist import *

import tensorflow as tf
mnist = read_data_sets('MNIST_data',one_hot = True)
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])

Y = tf.placeholder(tf.float32, shape=[None, 10])
#*******************************************************************
# DO NOT MODIFY THE CODE ABOVE THIS LINE
# MAKE CHANGES TO THE CODE BELOW:

# a method for initializing weights. Initialize to small random values
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.08)
    return tf.Variable(initial)

# a method for initializing bias. Initialize to 0.1
def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape)
  return tf.Variable(initial)

# do cross correlation with specifc strides and padding

def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# First Convolutional Layer.

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

X_image = tf.reshape(X, [-1,28,28,1]) # X_image ?x28x28x1
h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1) # h_conv1 ?x28x28x32
h_pool1 = max_pool_2x2(h_conv1) # h_pool1 ?x14x14x32


# Second Convolutional Layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # h_conv2 ?x14x14x64
h_pool2 = max_pool_2x2(h_conv2) # h_pool2 ?x7x7x64

# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # h_pool2_flat ?x7.7.64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # h_fc1 ?x1024

# Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # h_fc1_drop ?x1024


# Readout Layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # y_conv ?x10

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))

# regularization parameters
lambda_1 = 0
lambda_2 = 0
lambda_3 = 0
regularizer = (
  lambda_1*tf.nn.l2_loss(W_fc1) + 
  lambda_2*tf.nn.l2_loss(W_fc2)
  )

loss = tf.reduce_mean(cross_entropy + regularizer)

train_step = tf.train.AdamOptimizer(1.3e-3).minimize(loss)
sess.run(tf.global_variables_initializer())

print("Starting Training...")
print("epoch\ttrain_accuracy\ttest_accuracy")

for i in range(3000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_accuracy = accuracy.eval(feed_dict={
            X: mnist.test.images, 
            Y: mnist.test.labels,
            keep_prob: 1}) 
        train_accuracy = accuracy.eval(feed_dict={
            X: mnist.train.images, 
            Y: mnist.train.labels,
            keep_prob: 1})

        print("%d \t %.3f \t\t %.3f" % (i, train_accuracy, test_accuracy))
    # TRAIN STEP
    train_step.run(feed_dict={
             X: batch[0], 
             Y: batch[1], 
             keep_prob:0.5})
#end for loop

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = accuracy.eval(feed_dict={
            X: mnist.test.images, 
            Y: mnist.test.labels,
            keep_prob:1})

print("test accuracy: ", test_accuracy)
