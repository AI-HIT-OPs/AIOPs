import pandas as pd
import numpy as np


timesteps = 20


class Reader(object):

    def __init__(self, filename, column):
        self.dataFrame = pd.read_csv(filename)
        self.data = self.dataFrame[column].values.tolist()
        # Z-Score Normalization
        self.data = self.MaxMinNormalization(self.data,
                                             np.mean(self.data, axis=0),
                                             np.std(self.data, axis=0))

        self.train_x, self.train_y = self.generateData(self.data)
        self.min = np.min(self.data)
        self.max = np.max(self.data)

    def MaxMinNormalization(self, x, mean, std):
        x = (x - mean) / std
        return x

    def generateData(self, seq):
        """
        :param seq: continuous sequence of value
        :return:
        """
        x = []
        y = []
        for i in range(len(seq) - timesteps):
            x.append(seq[i: i + timesteps])
            y.append(seq[i + timesteps])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def getBatch(self, x, y, batch_size=128):
        for i in range(0, len(x), batch_size):
            begin_i = i
            end_i = i + batch_size if (i + batch_size) < len(x) else len(x)
            yield x[begin_i:end_i], y[begin_i:end_i]


if __name__ == "__main__":
    # test class
    reader = Reader("../datasets/db_os_stat(180123-180128).csv", "FREE_MEM_SIZE")
    print(reader.train_x)