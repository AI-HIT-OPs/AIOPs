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


# if __name__ == "__main__":
#     newdata = NewData(filename="../datasets/1-10.8.160.17_20181027_20181109.csv",
#                       column="AVAILABLE_MEMORY",
#                       timesteps=20, start=1, end=17280)
#     print(newdata.data[len(newdata.data) - 1])