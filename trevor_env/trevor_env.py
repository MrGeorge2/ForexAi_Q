from dataframe import dataframe
import cfg
import numpy as np
import time


class Trevor:
    def __init__(self, df: dataframe.Dataframe):
        self.__df = df

        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0

        self.total_reward = 0

    def reset(self):
        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0

    def step(self, action):
        sample, actual_ask, actual_bid = self.__df.get(self.cursor)
        sample = self.__append_last_action(sample=sample, action=action)

        reward = self.__process_action(action=action, actual_ask=actual_ask, actual_bid=actual_bid)
        self.__increment_cursor()

        return sample, reward

    def __process_action(self, action, actual_ask, actual_bid):
        if action < 0 or action > 2:
            raise ValueError(f'Action have to be inrage (0 - 2) got {action}')

        # """ CLOSING POSITION """
        if (self.last_action == 2 and action == 0) or (self.last_action == 1 and action == 0):
            reward = self.__close_trade(actual_bid=actual_bid, actual_ask=actual_ask)

        # """ CLOSING POSITION AND GOING TO DIFFERENT POSITION """
        elif (self.last_action == 2 and action == 1) or (self.last_action == 1 and action == 2):
            reward = self.__close_trade(actual_bid=actual_bid, actual_ask=actual_ask)
            self.enter_price = actual_ask if action == 2 else actual_bid

        # """ HOLDING OPENED POSITION  """
        elif (self.last_action == 2 and action == 2) or (self.last_action == 1 and action == 1):
            if self.last_action == 2:
                reward = (actual_ask - self.enter_price) * cfg.REWARD_FOR_PIPS

            else:
                reward = (self.enter_price - actual_bid) * cfg.REWARD_FOR_PIPS

        # """ OPENING POSITION  """
        elif (self.last_action == 0 and action == 1) or (self.last_action == 0 and action == 2):
            if action == 1:
                self.enter_price = actual_bid

            else:
                self.enter_price = actual_ask
            reward = cfg.HOLD_REWARD

        # """ HOLD """
        elif self.last_action == 0 and action == 0:
            reward = cfg.HOLD_REWARD

        else:
            raise ValueError(f'Last action = {self.last_action} and actual_action = {action}')

        self.last_action = action
        self.total_reward += reward
        return reward

    def __increment_cursor(self):
        """ Incrementing the cursor, if the cursor is bigger than lenght of the dataframe, then reset it"""

        self.cursor += 1
        if self.cursor > self.__df.lenght:
            self.reset()

    def __close_trade(self, actual_ask, actual_bid):
        if self.last_action == 2:
            reward = (actual_ask - self.enter_price) * cfg.REWARD_FOR_PIPS * cfg.TIMES_FACTOR

        else:
            reward = (self.enter_price - actual_bid) * cfg.REWARD_FOR_PIPS * cfg.TIMES_FACTOR
        return reward

    def __append_last_action(self, sample: np.ndarray, action: int):
        how_many = sample.shape[2]
        action_arr = (np.expand_dims(np.asarray([action for i in range(0, how_many)]), axis=0))

        return np.expand_dims(np.append(sample[0], action_arr, axis=0), axis=0)


if __name__ == '__main__':
    tr = Trevor(dataframe.Dataframe())
    for _ in range(100):
        start = time.time()
        print(tr.step(0))
        print(time.time() - start)
