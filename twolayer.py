# made with love and / for tensorflow
# Aditya Jain, IIT Indore


'''
	output in : out2.txt
	95% accuracy

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# loading data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

# variables and stuff required to build
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# first layer : 200 neurons
W1 = tf.Variable(tf.truncated_normal([28*28, 200], stddev = 0.1))
B1 = tf.Variable(tf.zeros([200]))

# second layer : 10 neurons
W2 = tf.Variable(tf.truncated_normal([200, 10], stddev = 0.1))
B2 = tf.Variable(tf.zeros([10]))

# variable initialisation
init = tf.global_variables_initializer()

# first layer : sigmoid, second layer : softmax
Tmp = tf.matmul(tf.reshape(X, [-1, 28*28]), W1) + B1
Y1 = tf.nn.sigmoid(Tmp)
Res = tf.matmul(Y1, W2) + B2
Y = tf.nn.softmax(Res)

# expected output (labels)
y = tf.placeholder(tf.float32, [None, 10])

# cross entropy and accuracy calculation
crossEntropy = -tf.reduce_sum(y*tf.log(Y))
isCorrect = tf.equal(tf.argmax(Y,1), tf.argmax(y,1))
accurate = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

# application of Gradient Descent (training)
optimizer = tf.train.GradientDescentOptimizer(0.003)
train = optimizer.minimize(crossEntropy)

# execute learning.
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_X, batch_Y = mnist.train.next_batch(100)
	sess.run(train, {X : batch_X, y : batch_Y})
	a, c = sess.run([accurate, crossEntropy], {X : mnist.test.images, y : mnist.test.labels})
	print(a,c)