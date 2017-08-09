import tensorflow as tf
from database import db

from settings import input_size_PCA, first_hidden_layer, hidden_layer_input

x = tf.placeholder(tf.float32, [None, input_size_PCA])

W1 = tf.Variable(
    tf.random_normal(shape=[input_size_PCA, first_hidden_layer], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None,
                     name=None))
b1 = tf.Variable(
    tf.random_normal(shape=[first_hidden_layer], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None))
y1 = hidden_layer_input(x, W1, b1)

W2 = tf.Variable(
    tf.random_normal(shape=[first_hidden_layer, 8], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None))
b2 = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None))

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


y = tf.matmul(y1, W2) + b2
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

for _ in range(1000):
    batch_xs, batch_ys = db.next_batch(500)

    if (_ % 100 == 0):
        print(0.5, sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.5})
    # if (_ % 200 == 0):
    #     sess.run(train_step, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001})
# print("valid", sess.run(cost, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001}))

for i in range(1000, 2000):
    batch_xs, batch_ys = db.next_batch(500)

    if (i % 100 == 0):
        print(0.1, sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.1})
    # if (_ % 200 == 0):
    #     sess.run(train_step, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001})
# print("valid", sess.run(cost, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001}))

for i in range(2000, 7000):
    batch_xs, batch_ys = db.next_batch(500)

    if (i % 100 == 0):
        print(0.01, sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.01})
    # if (_ % 200 == 0):
    #     sess.run(train_step, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001})
# print("valid", sess.run(cost, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001}))

for i in range(7000, 12000):
    batch_xs, batch_ys = db.next_batch(500)

    if (i % 100 == 0):
        print(0.001, sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: 0.001})
    # if (_ % 200 == 0):
    #     sess.run(train_step, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001})
# print("valid", sess.run(cost, feed_dict={x: db.valid_data, y_: db.valid_labels, learning_rate: 0.001}))

save_path = saver.save(sess, "/home/sharare/PycharmProjects/roboinstruct_training/states/last/model.ckpt")
error = cost
print(sess.run(error, feed_dict={x: db.test_data, y_: db.test_labels}))
