import tensorflow as tf
from database import db

x = tf.placeholder(tf.float32, [None, 15])
W = tf.Variable(tf.random_normal(shape=[15, 8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
b = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 8])

# cost = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * 50000)
cost = tf.reduce_mean(tf.square(y_ - y))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
train_step = tf.train.AdamOptimizer(0.5).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(10000):
    batch_xs = db.data
    batch_ys = db.labels
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

error = cost
print(sess.run(error, feed_dict={x: db.test_data, y_: db.test_labels}))
