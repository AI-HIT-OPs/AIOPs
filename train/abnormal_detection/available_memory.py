from utils.preprocessing import Data
import utils.lstm_model

import tensorflow as tf
import matplotlib.pyplot as plt

# settings of lstm model
timesteps = 20
batch_size = 64
epochs = 10

lstm_size = 30
lstm_layers = 2

filename = "../../datasets/db_os_stat(180123-180128).csv"
model = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
column = "FREE_MEM_SIZE"


def train_model():
    # prepare data
    data = Data(filename=filename, column=column, timesteps=timesteps)
    # init placeholder
    x, y, keep_prob = utils.lstm_model.initPlaceholder(timesteps)
    predictions = utils.lstm_model.lstm_model(x,
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
