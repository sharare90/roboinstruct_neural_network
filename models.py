import tensorflow as tf


def initialize_variable(shape, mean=0.0, stddev=0.25, dtype=tf.float32, seed=None, name=None):
    return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name))


class NeuralNetwork(object):
    def __init__(self, layers, learning_rate=0.001):
        self.learning_rate_value = learning_rate
        self.layers = layers

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.layers[0]))
        self.y_ = tf.placeholder(dtype=tf.float32, shape=(None, self.layers[-1]))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.W = []
        self.B = []

        for i in range(len(self.layers) - 1):
            w_i = initialize_variable(shape=(layers[i], layers[i + 1]))
            self.W.append(w_i)

            b_i = initialize_variable(shape=(1, layers[i + 1]))
            self.B.append(b_i)

        self.y = self.define_feed_forward()
        self.cost = self.define_cost()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def define_cost(self):
        return tf.reduce_mean(tf.square(self.y - self.y_))

    def define_feed_forward(self):
        output = self.x

        for i in range(len(self.W) - 1):
            layer_input = 2. / 3. * tf.matmul(output, self.W[i]) + self.B[i]
            output = 1.7159 * tf.tanh(layer_input)

        return tf.matmul(output, self.W[-1]) + self.B[-1]

    def feed_forward(self, x):
        self.sess.run(self.y, feed_dict={self.x: x})

    def evaluate_cost(self, x, y):
        self.sess.run(self.cost, feed_dict={self.x: x, self.y_: y})

    def train(self, x_train, y_train, iterations):
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        for _ in range(iterations):
            batch_xs, batch_ys = x_train, y_train

            if _ % 100 == 0:
                print(self.evaluate_cost(batch_xs, batch_ys))

            self.sess.run(train_step, feed_dict={
                self.x: batch_xs, self.y_: batch_ys, self.learning_rate: self.learning_rate_value
            })

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
