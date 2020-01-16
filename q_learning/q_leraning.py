# -*- coding: utf-8 -*-
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
import os
from dataframe import dataframe
from trevor_env import trevor_env
import cfg
from threading import Thread


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.97  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.batch_size = batch_size
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
        model.add(CuDNNLSTM(units=48, return_sequences=True, input_shape=self.state_size))
        model.add(Dropout(0.2))

        model.add(CuDNNLSTM(units=32, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(units=12, activation='relu'))

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
        # return 0, True
        if not isinstance(state, np.ndarray):
            return 0

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True
        act_values = self.model.predict(state, steps=1)
        return np.argmax(act_values[0]), False  # returns action

    def predict(self, state):
        act_values = self.model.predict(state, steps=1)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        while True:
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                if not isinstance(state, np.ndarray):
                    continue

                target = self.model.predict(state, steps=1, verbose=0)
                if done and reward > 80 * cfg.TIMES_FACTOR:
                    target[0][action] = reward
                else:
                    # a = self.model.predict(next_state)[0]
                    t = self.target_model.predict(next_state)[0]
                    target[0][action] = reward + self.gamma * np.amax(t)
                    # target[0][action] = reward + self.gamma * t[np.argmax(a)]
                self.model.fit(state, target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            # print('done')

    def load(self, name):
        self.model.load_weights(name)
        self.model._make_predict_function()
        self.model._make_test_function()
        self.model._make_train_function()

        self.target_model.load_weights(name)
        self.target_model._make_predict_function()
        self.target_model._make_test_function()
        self.target_model._make_train_function()

    def save(self, name):
        self.model.save_weights(name)


def eval_test(state_size, action_size):
    envv = trevor_env.Trevor(dataframe.Dataframe())

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
    env = trevor_env.Trevor(dataframe.Dataframe())
    state_size = (cfg.NUMBER_OF_SAMPLES, 9)
    action_size = 3
    batch_size = 32
    agent = DQNAgent(state_size, action_size, batch_size)

    # agent.save("./save/cartpole-ddqn.h5")
    agent.load("./save/cartpole-ddqn.h5")

    closed = False
    run = False

    for e in range(cfg.EPISODES):
        state = env.reset()

        for time in range(env.df.lenght):
            action, random_action = agent.act(state)

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

            print(f'Actual reward = {round(reward, 1)},\t total reward = {round(env.total_reward, 1)},'
                  f'\t action = {action}, \t trade_counter = {round(env.trade_counter, 1)}, '
                  f'\t pip_counter = {round(env.closed_counter, 1)}'
                  f'\t random_action = {random_action}'
                  f'\t candle_number = {time}')

            if closed and reward > 80 * cfg.TIMES_FACTOR:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e, cfg.EPISODES, time, round(agent.epsilon, 2)))

            if len(agent.memory) > batch_size:
                # agent.replay(batch_size)
                if not run:
                    thr_list = [Thread(target=agent.replay) for _ in range(15)]
                    for thr in thr_list:
                        thr.start()
                        t_lib.sleep(1)
                    run = True
        env.plot(title=f'total reward ={env.total_reward};  e = {round(agent.epsilon, 2)}')
        env.reset_closed_list()
        agent.save("./save/cartpole-ddqn.h5")