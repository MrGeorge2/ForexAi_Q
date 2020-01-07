import os
import time
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
from keras.optimizers import Adam
from keras import backend as K
import time as t_lib
import tensorflow as tf


DATAFRAME_NAME = 'EURUSD_m15_Ask_ready.csv'
NUMBER_OF_SAMPLES = 100

EPISODES = 5000
TICQTY_MAX = 55000
HOLD_REWARD = -0.1
REWARD_FOR_PIPS = 10_000
TIMES_FACTOR = 1.5

ACTION_DECODE = {
    0: 0,
    1: 0.5,
    2: 1,
}


class Dataframe:

    def __init__(self):
        self._dataframe = self._load()
        self.__scaler = MinMaxScaler()

    @property
    def lenght(self):
        return len(self._dataframe.index) - NUMBER_OF_SAMPLES

    def get(self, sample_number):
        if sample_number > self.lenght or sample_number < 0:
            raise ValueError(f"Sample number out of range (0 - {self.lenght})")

        start_index = sample_number
        end_index = start_index + NUMBER_OF_SAMPLES

        df_sample = self._dataframe[start_index: end_index]

        last_open = df_sample.at[df_sample.index[-1], 'open']
        last_close = df_sample.at[df_sample.index[-1], 'close']

        df_sample = df_sample[['open', 'close', 'high', 'low', 'tickqty']].values
        df_sample = self._scale(df_sample, start=0, end=4)
        return np.expand_dims(df_sample, axis=0), last_open, last_close

    @staticmethod
    def _load():
        """ Creating relative path and then loading the df_path """
        df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) +
                               os.path.normpath('/dfs/{}'.format(DATAFRAME_NAME)))
        df = pd.read_csv(
            df_path,
            dtype={
                'datetime'
                'open': np.float32,
                'close': np.float32,
                'high': np.float32,
                'low': np.float32,
                'tickqty': np.float32,
            }
        )

        df['tickqty'] = df['tickqty'] / TICQTY_MAX
        return df

    def _scale(self, array: np.ndarray, start: int, end: int):
        columns = array.T[start: end].T

        self.__scaler.fit(columns)
        scaled_cols = self.__scaler.transform(columns).T
        array.T[start:end] = scaled_cols
        return array


class Trevor:
    def __init__(self, df):
        self.df = df

        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0

        self.closed_counter = 0
        self.total_reward = 0
        self.trade_counter = 0

    def reset(self):
        self.cursor = 0
        self.enter_price = 0
        self.last_action = 0
        self.closed_counter = 0
        self.trade_counter = 0

    def step(self, action):
        sample, last_open, last_close = self.df.get(self.cursor)
        sample = self.__append_last_action(sample=sample, action=action)

        reward, closing_trade = self.__process_action(action=action, last_open=last_open, last_close=last_close)
        self.__increment_cursor()

        return sample, reward, closing_trade, ''

    def get_total_reward(self):
        return self.total_reward

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
            closing_trade = True

        # """ HOLDING OPENED POSITION  """
        elif (self.last_action == 2 and action == 2) or (self.last_action == 1 and action == 1):
            if self.last_action == 2:
                reward = (last_close - self.enter_price) * REWARD_FOR_PIPS

            else:
                reward = (self.enter_price - last_close) * REWARD_FOR_PIPS

        # """ OPENING POSITION  """
        elif (self.last_action == 0 and action == 1) or (self.last_action == 0 and action == 2):
            if action == 1:
                self.enter_price = last_open

            else:
                self.enter_price = last_open
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

    def __close_trade(self, last_close):
        if self.last_action == 2:
            reward = (last_close - self.enter_price) * REWARD_FOR_PIPS * TIMES_FACTOR
            self.closed_counter += reward / TIMES_FACTOR
            reward += 0.00001 * pow(reward, 3)

        else:
            reward = (self.enter_price - last_close) * REWARD_FOR_PIPS * TIMES_FACTOR
            self.closed_counter += reward / TIMES_FACTOR
            reward += 0.00001 * pow(reward, 3)

        self.trade_counter += 1
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
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
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
        model.add(CuDNNLSTM(units=32, return_sequences=True, input_shape=self.state_size))
        model.add(Dropout(0.2))

        model.add(CuDNNLSTM(units=16, return_sequences=False))
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
        if not isinstance(state, np.ndarray):
            return 0

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, steps=1)
        return np.argmax(act_values[0])  # returns action

    def predict(self, state):
        act_values = self.model.predict(state, steps=1)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not isinstance(state, np.ndarray):
                continue

            target = self.model.predict(state, steps=1, verbose=0)
            if done and reward > 20:
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


def eval_test(state_size, action_size):
    envv = Trevor(Dataframe())

    agentt = DQNAgent(state_size, action_size)
    agentt.load("./save/cartpole-ddqn.h5")

    sample = envv.reset()

    for i in range(envv.df.lenght):
        acc = agentt.predict(sample)
        sample, rewardd, closedd, _ = envv.step(acc)
        print('Actual reward = {},\t total reward = {},\t action = {}'.format(round(rewardd, 3),
                                                                              round(envv.get_total_reward(), 3),
                                                                              acc))


if __name__ == "__main__":
    env = Trevor(Dataframe())
    state_size = (NUMBER_OF_SAMPLES, 6)
    action_size = 3
    agent = DQNAgent(state_size, action_size)

    agent.load("./save/cartpole-ddqn.h5")

    closed = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()

        for time in range(env.df.lenght):
            action = agent.act(state)

            if action > 3 or action < 0:
                print('Got action ' + action)
                continue

            next_state, reward, closed, _ = env.step(action)

            if not isinstance(next_state, np.ndarray) or not(state, np.ndarray):
                print(next_state)
                print('NOT NUMPY!!')
                continue

            agent.memorize(state=state, action=action, reward=reward, next_state=next_state, done=closed)
            state = next_state

            print(f'Actual reward = {round(reward, 2)},\t total reward = {round(env.total_reward, 2)},'
                  f'\t action = {action}, \t trade_counter = {round(env.trade_counter, 2)}, '
                  f'\t pip_counter = {env.closed_counter}')

            if closed and reward > 0:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e, EPISODES, time, agent.epsilon))

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.save("./save/cartpole-ddqn.h5")
