from dataframe import dataframe
import cfg
import numpy as np
import time
from math import pow
from matplotlib import pyplot


class Trevor:
    POSITIVE_TIMES_REWARD = 0
    NEGATIVE_TIMES_REWARD = 0

    def __init__(self, df: dataframe.Dataframe):
        self.df = df

        self.cursor = 0
        self.enter_price = 0
        self.local_max_price = 0

        self.last_action = 0

        self.closed_counter = 0
        self.total_reward = 0
        self.trade_counter = 0

        self.closed_counter_list = []

    def reset(self):
        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0
        self.closed_counter = 0
        self.trade_counter = 0
        self.total_reward = 0
        # self.reset_closed_list()

        return self.step(0)[0]

    def step(self, action):
        sample, last_open, last_close = self.df.get(self.cursor)
        sample = self.__append_last_action(sample=sample, action=action)

        reward, closing_trade = self.__process_action(action=action, last_open=last_open, last_close=last_close)
        self.__increment_cursor()

        return sample, reward, closing_trade, ''

    def get_total_reward(self):
        return self.total_reward

    def reset_closed_list(self):
        self.closed_counter_list = []

    def plot(self, title):
        x = list(range(1, len(self.closed_counter_list) + 1))
        pyplot.plot(x, self.closed_counter_list)
        pyplot.title(str(title))
        pyplot.show()

    def __process_action(self, action, last_open, last_close):
        if action < 0 or action > 2:
            raise ValueError('Action have to be inrage (0 - 2) got {action}')

        closing_trade = False

        # """ CLOSING POSITION """
        if (self.last_action == 2 and action == 0) or (self.last_action == 1 and action == 0):
            reward = self.__close_trade(last_close=last_close)
            closing_trade = True

        # """ CLOSING POSITION AND GOING TO DIFFERENT POSITION """
        elif (self.last_action == 2 and action == 1) or (self.last_action == 1 and action == 2):
            reward = self.__close_trade(last_close=last_close)
            self.enter_price = last_open
            self.local_max_price = last_open
            closing_trade = True

        # """ HOLDING OPENED POSITION  """
        elif (self.last_action == 2 and action == 2) or (self.last_action == 1 and action == 1):
            if self.last_action == 2:
                if self.local_max_price < last_close:
                    reward = (last_close - self.enter_price) * cfg.REWARD_FOR_PIPS
                    self.local_max_price = last_close

                else:
                    reward = (last_close - self.local_max_price) * cfg.REWARD_FOR_PIPS
                    reward = reward / cfg.DIVIDE_PRICE_UNDER_LOCAL_MINIMA if self.local_max_price > self.enter_price \
                        else reward

            else:
                if self.local_max_price > last_close:
                    reward = (self.enter_price - last_close) * cfg.REWARD_FOR_PIPS
                    self.local_max_price = last_close

                else:
                    reward = (self.local_max_price - last_close) * cfg.REWARD_FOR_PIPS
                    reward = reward / cfg.DIVIDE_PRICE_UNDER_LOCAL_MINIMA if self.local_max_price < self.enter_price \
                        else reward

        # """ OPENING POSITION  """
        elif (self.last_action == 0 and action == 1) or (self.last_action == 0 and action == 2):
            if action == 1:
                self.enter_price = last_open

            else:
                self.enter_price = last_open
            reward = cfg.HOLD_REWARD

        # """ HOLD """
        elif self.last_action == 0 and action == 0:
            reward = cfg.HOLD_REWARD

        else:
            raise ValueError('Last action = {self.last_action} and actual_action = {action}')

        self.last_action = action
        self.total_reward += reward
        return reward, closing_trade

    def __increment_cursor(self):
        """ Incrementing the cursor, if the cursor is bigger than lenght of the dataframe, then reset it"""

        self.cursor += 1
        if self.cursor > self.df.lenght:
            self.reset()

    def __close_trade(self, last_close):
        if self.last_action == 2:
            reward = (last_close - self.enter_price) * cfg.REWARD_FOR_PIPS * cfg.TIMES_FACTOR
            self.closed_counter += reward / cfg.TIMES_FACTOR
            reward += self.POSITIVE_TIMES_REWARD * pow(reward, 3) if reward > 0 \
                else self.NEGATIVE_TIMES_REWARD * pow(reward, 3)

        else:
            reward = (self.enter_price - last_close) * cfg.REWARD_FOR_PIPS * cfg.TIMES_FACTOR
            self.closed_counter += reward / cfg.TIMES_FACTOR
            reward += self.POSITIVE_TIMES_REWARD * pow(reward, 3) if reward > 0 \
                else self.NEGATIVE_TIMES_REWARD * pow(reward, 3)

        self.closed_counter_list.append(self.closed_counter)
        self.trade_counter += 1
        return reward

    def __append_last_action(self, sample: np.ndarray, action: int):
        how_many = sample.shape[1]
        action = cfg.ACTION_DECODE[action]

        action_arr = (np.expand_dims(np.asarray([action for i in range(0, how_many)]), axis=1))

        return np.expand_dims(np.append(sample[0], action_arr, axis=1), axis=0)


if __name__ == '__main__':
    tr = Trevor(dataframe.Dataframe())
    for _ in range(100):
        start = time.time()
        print(tr.step(0))
        print(time.time() - start)
