import pandas as pd
import numpy as np


class Reader(object):

    def __init__(self, filename, column):
        self.dataframe = pd.read_csv(filename)
        self.data = self.dataframe.loc[:, [column]]
        self.data = np.array(self.data)
        self.data = self.MaxMinNormalization(self.data,
                                             np.mean(self.data, axis=0),
                                             np.std(self.data, axis=0))

    def MaxMinNormalization(self, x, mean, std):
        x = (x - mean) / std
        return x


if __name__ == "__main__":
    # test class
    reader = Reader("../datasets/db_os_stat(180123-180128).csv", "FREE_MEM_SIZE")
    print(reader.data)