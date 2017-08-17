# made with love and / for tensorflow
# Aditya Jain, IIT Indore

'''
	output in : fivelayerC5000.txt
	98% accuracy

'''

import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

# loading data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

# variables, weights and biases for nn
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

W1 = tf.Variable(tf.truncated_normal([784, 200], stddev = 0.1))
B1 = tf.Variable(tf.ones([200])/10)

W2 = tf.Variable(tf.truncated_normal([200, 100], stddev = 0.1))
B2 = tf.Variable(tf.ones([100])/10)

W3 = tf.Variable(tf.truncated_normal([100, 60], stddev = 0.1))
B3 = tf.Variable(tf.ones([60])/10)

W4 = tf.Variable(tf.truncated_normal([60, 30], stddev = 0.1))
B4 = tf.Variable(tf.ones([30])/10)

W5 = tf.Variable(tf.truncated_normal([30, 10], stddev = 0.1))
B5 = tf.Variable(tf.ones([10])/10)

# five layers : 4 relu and 1 softmax
Y1 = tf.nn.relu(tf.matmul(tf.reshape(X, [-1, 784]), W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# desired output (labels)
y = tf.placeholder(tf.float32, [None, 10])

# cross entropy and accuracy calculation
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))*100
isCorrect = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
accurate = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

lr = tf.placeholder(tf.float32, [])

# application of Adam Optimizer
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(crossEntropy)

# variable initialization
init = tf.global_variables_initializer()

# execute learning.
sess = tf.Session()
sess.run(init)

# setting for exponential decay of learning rate from 0.003 to 0.0001
lrx = 0.003
lrm = 0.0001

for i in range(5000):
	batch_X, batch_Y = mnist.train.next_batch(100)
	dlr = lrm + (lrx - lrm)*math.exp(-i/2000)
	sess.run(train, {X : batch_X, y : batch_Y, lr : dlr})
	a, c = sess.run([accurate, crossEntropy], {X : mnist.test.images, y : mnist.test.labels, lr : dlr})
	print(a,c)