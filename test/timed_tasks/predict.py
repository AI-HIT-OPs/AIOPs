import tensorflow as tf
import numpy as np
import pandas as pd


class Data(object):

    def __init__(self, filename, column, timesteps):
        self.timesteps = timesteps
        self.filename = filename
        self.column = column
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
        data = data[self.column].values.tolist()
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

    def getFakeData(self, x, batch_size=1):
        begin_i = 0
        end_i = batch_size if batch_size < len(x) else len(x)
        return x[begin_i:end_i]


class Predict:
    def __init__(self, model):
        # set GPU memory limitation
        gpu_options = tf.GPUOptions(allow_growth=True)
        # initial session
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model + ".meta")
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model)
                self.predictions = tf.get_collection('predictions')[0]
                self.graph = tf.get_default_graph()
                # get input frame placeholder by operation name
                self.x = self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]

    def predict(self, filename, column, timesteps):
        data = Data(filename=filename, column=column, timesteps=timesteps)
        test_data = data.getFakeData(data.train_x, batch_size=1)
        feed_dict = {self.x: test_data[:, :, None], self.keep_prob: 1.0}
        results = self.sess.run(self.predictions, feed_dict=feed_dict)
        print(results)


if __name__ == "__main__":
    # settings
    timesteps = 20
    batch_size = 64

    filename_available_memory = "../../datasets/db_os_stat(180123-180128).csv"
    filename_consistent_gets = "../../datasets/db_stat(180123-180128).csv"
    model_available_memory = "../../model/abnormal_detection_model/available_memory_model/FREE_MEM_SIZE_MODEL"
    model_consistent_gets = "../../model/abnormal_detection_model/consistent_gets_model/CONSISTENT_GETS_MODEL"
    column_available_memory = "FREE_MEM_SIZE"
    column_consistent_gets = "CONSISTENT_GETS"
    # multi model test
    predict1 = Predict(model_available_memory)
    predict1.predict(filename_available_memory, column_available_memory, timesteps)
    predict2 = Predict(model_consistent_gets)
    predict2.predict(filename_consistent_gets, column_consistent_gets, timesteps)
