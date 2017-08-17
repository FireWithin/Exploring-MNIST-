# made with love and / for tensorflow
# Aditya Jain, IIT Indore

'''
	output in : fivelayerA.txt
	NaN problem
	does not work correctly.

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# loading data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

# variables, weights and biases for nn
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

W1 = tf.Variable(tf.truncated_normal([784, 200], stddev = 0.1))
B1 = tf.Variable(tf.zeros([200]))

W2 = tf.Variable(tf.truncated_normal([200, 100], stddev = 0.1))
B2 = tf.Variable(tf.zeros([100]))

W3 = tf.Variable(tf.truncated_normal([100, 60], stddev = 0.1))
B3 = tf.Variable(tf.zeros([60]))

W4 = tf.Variable(tf.truncated_normal([60, 30], stddev = 0.1))
B4 = tf.Variable(tf.zeros([30]))

W5 = tf.Variable(tf.truncated_normal([30, 10], stddev = 0.1))
B5 = tf.Variable(tf.zeros([10]))

# five layers : 4 sigmoid and 1 softmax
Y1 = tf.nn.sigmoid(tf.matmul(tf.reshape(X, [-1, 784]), W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

# desired output (labels)
y = tf.placeholder(tf.float32, [None, 10])

# cross entropy and accuracy calculation
crossEntropy = tf.reduce_sum(y * tf.log(Y))
isCorrect = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
accurate = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

# application of gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.003)
train = optimizer.minimize(crossEntropy)

# variable initialization
init = tf.global_variables_initializer()

# execute learning.
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_X, batch_Y = mnist.train.next_batch(100)
	sess.run(train, {X : batch_X, y : batch_Y})
	a, c = sess.run([accurate, crossEntropy], {X : mnist.test.images, y : mnist.test.labels})
	print(a,c)