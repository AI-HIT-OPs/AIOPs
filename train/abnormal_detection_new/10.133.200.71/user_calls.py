import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# settings of lstm model
timesteps = 20
batch_size = 64
epochs = 5

lstm_size = 30
lstm_layers = 2

filename = "../../../datasets/1-10.133.200.71_20181027_20181109.csv"
model = "../../../model/abnormal_detection_model_new/10.133.200.71/user_calls_model/USER_CALLS_MODEL"
column = "USER_CALLS"
start = 362816
end = 380095


class NewData(object):
    def __init__(self, filename, column, timesteps, start, end):
        self.timesteps = timesteps
        self.filename = filename
        self.column = column
        self.start = start
        self.end = end
        self.train_x, self.train_y, self.test_x, self.test_y = self.preprocess()

    def MaxMinNormalization(self, x, max_value, min_value):
        """
        :param x:           data
        :param max_value:   max value in the data
        :param min_value:   min value in the data
        :return:            normalization data
        """
        x = (x - min_value) / (max_value - min_value)
        return x

    def generateGroupDataList(self, seq):
        """
        :param seq:         continuous sequence of value in data
        :return:            input data array and label data array in the format of numpy
        """
        x = []
        y = []
        for i in range(len(seq) - self.timesteps):
            x.append(seq[i: i + self.timesteps])
            y.append(seq[i + self.timesteps])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def preprocess(self):
        """
        :return:            training data and testing data of given filename and column
        """
        data = pd.read_csv(self.filename)
        data = data["VALUE"].values.tolist()
        data = data[self.start - 1:self.end]

        data = self.MaxMinNormalization(data,
                                        np.max(data, axis=0),
                                        np.min(data, axis=0))

        train_x, train_y = self.generateGroupDataList(data)
        test_x, test_y = self.generateGroupDataList(data)

        return train_x, train_y, test_x, test_y

    def getBatches(self, x, y, batch_size):
        for i in range(0, len(x), batch_size):
            begin_i = i
            end_i = i + batch_size if (i + batch_size) < len(x) else len(x)
            yield x[begin_i:end_i], y[begin_i:end_i]


def initPlaceholder(timesteps):
    x = tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return x, y_, keep_prob


def lstm_model(x, lstm_size, lstm_layers, keep_prob):
    # define basis structure LSTM cell
    def lstm_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    # multi layer LSTM cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    # dynamic rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    # reverse
    outputs = outputs[:, -1]
    # fully connected
    predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.sigmoid)
    return predictions


def train_model():
    # prepare data
    data = NewData(filename=filename, column=column, timesteps=timesteps, start=start, end=end)
    # init placeholder
    x, y, keep_prob = initPlaceholder(timesteps)
    predictions = lstm_model(x,
                             lstm_size=lstm_size,
                             lstm_layers=lstm_layers,
                             keep_prob=keep_prob)
    # mse loss function
    cost = tf.losses.mean_squared_error(y, predictions)
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    tf.add_to_collection("predictions", predictions)
    saver = tf.train.Saver()

    # define session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        # batches counter
        iteration = 1
        # loop for training
        for epoch in range(epochs):
            for xs, ys in data.getBatches(data.train_x, data.train_y, batch_size):
                feed_dict = {x: xs[:, :, None], y: ys[:, None], keep_prob: .5}
                loss, train_step = sess.run([cost, optimizer], feed_dict=feed_dict)

                if iteration % 100 == 0:
                    print('Epochs:{}/{}'.format(epoch, epochs),
                          'Iteration:{}'.format(iteration),
                          'Train loss: {}'.format(loss))
                iteration += 1
        # save model as checkpoint format to optional folder
        saver.save(sess, model)
        # test model
        feed_dict = {x: data.test_x[:, :, None], keep_prob: 1.0}
        results = sess.run(predictions, feed_dict=feed_dict)
        plt.plot(results, 'r', label='predicted')
        plt.plot(data.test_y, 'g--', label='real')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    train_model()
