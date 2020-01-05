# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras import backend as K
import time as t_lib
import tensorflow as tf
import os


EPISODES = 5000
DATAFRAME_NAME = 'EURUSD-2019-11.csv'
NUMBER_OF_SAMPLES = 10000

HOLD_REWARD = -0.1
REWARD_FOR_PIPS = 10000
TIMES_FACTOR = 10

ACTION_DECODE = {
    0: 0,
    1: 0.5,
    2: 1,
}


class Dataframe:

    def __init__(self):
        self._dataframe = self._load()

    @property
    def lenght(self):
        return len(self._dataframe.index) - NUMBER_OF_SAMPLES

    def get(self, sample_number):
        if sample_number > self.lenght or sample_number < 0:
            raise ValueError("Sample number out of range (0 - {self.lenght})")

        start_index = sample_number
        end_index = start_index + NUMBER_OF_SAMPLES

        df_sample = self._dataframe[start_index: end_index]

        actual_ask = df_sample.at[df_sample.index[-1], 'ask']
        actual_bid = df_sample.at[df_sample.index[-1], 'bid']

        return np.expand_dims(df_sample[['hours', 'minutes', 'microsec', 'bid', 'ask']].values, axis=0), \
               actual_ask, actual_bid

    @staticmethod
    def _load():
        """ Creating relative path and then loading the df_path """
        df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) +
                               os.path.normpath('/{}'.format(DATAFRAME_NAME)))
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


class Trevor:
    def __init__(self, df):
        self.df = df

        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0

        self.total_reward = 0

    def reset(self):
        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0

    def step(self, action):
        sample, actual_ask, actual_bid = self.df.get(self.cursor)
        sample = self.__append_last_action(sample=sample, action=action)

        reward, closing_trade = self.__process_action(action=action, actual_ask=actual_ask, actual_bid=actual_bid)
        self.__increment_cursor()

        return sample, reward, closing_trade, ''

    def __process_action(self, action, actual_ask, actual_bid):
        if action < 0 or action > 2:
            raise ValueError('Action have to be inrage (0 - 2) got {action}')

        closing_trade = False

        # """ CLOSING POSITION """
        if (self.last_action == 2 and action == 0) or (self.last_action == 1 and action == 0):
            reward = self.__close_trade(actual_bid=actual_bid, actual_ask=actual_ask)
            closing_trade = True

        # """ CLOSING POSITION AND GOING TO DIFFERENT POSITION """
        elif (self.last_action == 2 and action == 1) or (self.last_action == 1 and action == 2):
            reward = self.__close_trade(actual_bid=actual_bid, actual_ask=actual_ask)
            self.enter_price = actual_ask if action == 2 else actual_bid
            closing_trade = True

        # """ HOLDING OPENED POSITION  """
        elif (self.last_action == 2 and action == 2) or (self.last_action == 1 and action == 1):
            if self.last_action == 2:
                reward = (actual_ask - self.enter_price) * REWARD_FOR_PIPS

            else:
                reward = (self.enter_price - actual_bid) * REWARD_FOR_PIPS

        # """ OPENING POSITION  """
        elif (self.last_action == 0 and action == 1) or (self.last_action == 0 and action == 2):
            if action == 1:
                self.enter_price = actual_bid

            else:
                self.enter_price = actual_ask
            reward = HOLD_REWARD

        # """ HOLD """
        elif self.last_action == 0 and action == 0:
            reward = HOLD_REWARD

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

    def __close_trade(self, actual_ask, actual_bid):
        if self.last_action == 2:
            reward = (actual_ask - self.enter_price) * REWARD_FOR_PIPS * TIMES_FACTOR

        else:
            reward = (self.enter_price - actual_bid) * REWARD_FOR_PIPS * TIMES_FACTOR
        return reward

    def __append_last_action(self, sample: np.ndarray, action: int):
        how_many = sample.shape[1]
        action = ACTION_DECODE[action]

        action_arr = (np.expand_dims(np.asarray([action for i in range(0, how_many)]), axis=1))

        return np.expand_dims(np.append(sample[0], action_arr, axis=1), axis=0)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=self.state_size))
        model.add(Dropout(0.2))

        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not isinstance(state, np.ndarray):
                continue

            target = self.model.predict(state, steps=1, verbose=0)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = Trevor(Dataframe())
    state_size = (NUMBER_OF_SAMPLES, 6)
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    # agent.load("./save/cartpole-ddqn.h5")

    closed = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()

        for time in range(env.df.lenght):
            action = agent.act(state)
            next_state, reward, closed, _ = env.step(action)

            if not isinstance(next_state, np.ndarray) or not(state, np.ndarray):
                print(next_state)
                print('NOT NUMPY!!')
                continue

            agent.memorize(state=state, action=action, reward=reward, next_state=next_state, done=closed)
            state = next_state

            print('Actual reward = {},\t total reward = {},\t action = {}'.format(round(reward, 3), round(env.total_reward, 3), action))

            if closed and reward > 0:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.save("./save/cartpole-ddqn.h5")