import pandas as pd
import cfg
import os
import time
import numpy as np
import datetime


class Dataframe:

    def __init__(self):
        self._dataframe = self._load()

    @property
    def lenght(self):
        return len(self._dataframe.index) - cfg.NUMBER_OF_SAMPLES

    def get(self, sample_number):
        if sample_number > self.lenght or sample_number < 0:
            raise ValueError(f"Sample number out of range (0 - {self.lenght})")

        start_index = sample_number
        end_index = start_index + cfg.NUMBER_OF_SAMPLES

        df_sample = self._dataframe[start_index: end_index]

        actual_ask = df_sample.at[df_sample.index[-1], 'ask']
        actual_bid = df_sample.at[df_sample.index[-1], 'bid']

        return np.expand_dims(df_sample[['hours', 'minutes', 'microsec', 'bid', 'ask']].values, axis=0), \
               actual_ask, actual_bid

    @staticmethod
    def _load():
        """ Creating relative path and then loading the df_path """
        df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) +
                               os.path.normpath(f'/dfs/{cfg.DATAFRAME_NAME}'))
        df = pd.read_csv(
            df_path,
            names=[
                'currency_pair',
                'datetime',
                'bid',
                'ask'
            ],
            dtype={
                'datetime'
                'bid': np.float32,
                'ask': np.float32,
            }
        )
        df['hours'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S.%f').dt.hour / 24
        df['minutes'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S.%f').dt.minute / 64
        df['microsec'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H:%M:%S.%f').dt.microsecond / 1000000
        return df


if __name__ == '__main__':
    dff = Dataframe()

    for i in range(0, dff.lenght):
        start = time.time()
        val, ask, bid = dff.get(i)
        end = time.time()

        print(end - start)
