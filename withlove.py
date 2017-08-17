# Made with love and / for tensorflow
# Aditya Jain, IIT Indore


'''
	output in : out.txt
	91% accuracy

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

# variables and stuff required to build the model
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# variable initialisation
init = tf.global_variables_initializer()

# realisation of model, calculation of Y : model for estimation
Res = tf.matmul(tf.reshape(X, [-1, 28*28]), W) + b
Y = tf.nn.softmax(Res)

# actual (desired) output
y = tf.placeholder(tf.float32, [None, 10])

# cross entropy calculation
crossEntropy = -tf.reduce_sum(y * tf.log(Y))

# for checking accuracy of the network
isCorrect = tf.equal(tf.argmax(Y, 1), tf.argmax(y,1))
accurate = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

# application of gradient descent (training)
optimizer = tf.train.GradientDescentOptimizer(0.003)
train = optimizer.minimize(crossEntropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_X, batch_Y = mnist.train.next_batch(100)
	sess.run(train, {X : batch_X, y : batch_Y})
	a, c = sess.run([accurate, crossEntropy], {X : mnist.test.images, y : mnist.test.labels})
	print(a,c)