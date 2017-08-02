import tensorflow as tf
from database import db

x = tf.placeholder(tf.float32, [None, 15])
W = tf.Variable(tf.random_normal(shape=[15, 8], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None))
b = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None))

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 8])
learning_rate = tf.placeholder(tf.float32, shape=[])
# cost = tf.reduce_sum(tf.pow(y - y_, 2)) / (2 * len(db.data))
cost = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_xs = db.data
batch_ys = db.labels
for _ in range(100):
    # batch_xs, batch_ys = db.next_batch(100)
    # batch_xs = db.data
    # batch_ys = db.labels
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.5})
for i in range(100, 1000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.1})

for i in range(1000, 5000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.01})
for i in range(5000, 10000):
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.001})

save_path = saver.save(sess, "/home/sharare/PycharmProjects/roboinstruct_training/states/last/model.ckpt")
error = cost
print(sess.run(error, feed_dict={x: db.test_data, y_: db.test_labels}))
