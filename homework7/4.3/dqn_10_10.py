# -*- coding: utf-8 -*-

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20191109"

import random
import numpy as np
import json
import matplotlib.pyplot as plt
import pdb
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras.layers import Dense, Activation, Flatten

np.random.seed(2019)

"define the environment"
class Environment:
    def __init__(self):
        # set the reward matrix
        reward = np.load("reward_10_10.npy")

        self.max_row, self.max_col = reward.shape
        self.raw_reward = reward
        self.reward = reward
        self.action_map = {0: [0,-1], # left
                           1: [0,1],  # right
                           2: [-1,0], # up
                           3: [1,0],  # down
                           }

    def step(self, action, state):
        """Given an action, outputs the reward.
        Args:
            action: int, an action taken by the agent
            state: str, the current state of the agent

        Outputs:
            next_state: list, next state of the agent
            reward: float, the reward at the next state
            done: bool, stop or continue learning
        """
        done = False
        next_state = json.loads(state)
        state_shift = self.action_map[action]

        next_state = np.array(next_state) + np.array(state_shift)
        next_state = next_state.tolist()

        reward = self.reward[next_state[0], next_state[1]]
        self.reward[next_state[0], next_state[1]] = 0

        next_state = json.dumps(next_state)

        if reward < 0 or reward == 4:
            done = True

        return next_state, reward, done

    def reset(self, is_training=True):
        """Reset the environment, state, and reward matrix.
        Args:
            is_training: bool, if True, the initial state of the agent will be set randomly,
                or the initial state will be (0,0) by default.

        Outputs:
            state: str, the initial state.
        """
        if is_training:
            # randomly init
            while True:
                init_state = np.random.randint(0,10,(2))
                if self.reward[init_state[0],init_state[1]] >= 0:
                    init_state = init_state.tolist()
                    break
        else:
            init_state = [0,0]

        self.state = json.dumps(init_state)
        self.reward = self.raw_reward.copy()
        return self.state

"define the agent"
class Agent:
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs["gamma"]
        self.alpha = kwargs["alpha"]
        self.eps = kwargs["eps"]
        self.max_col = kwargs["max_col"]
        self.max_row = kwargs["max_row"]
        # action is [0,1,2,3]: left, right, up, down
        self.action_space = [0, 1, 2, 3]

        # DQN related parameters
        self.state_size = 2  # state dimension
        self.action_size = 4 # action size = 4
        self.memory = deque(maxlen=500)
        self.update_target_freq = 10
        self.update_model_freq = 10
        self.batch_size = 50
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=2, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        print("model saved:{}".format(name))

    def do(self, state):
        """Know the current state, choose an action.
        Args:
            state: str, as "[0, 0]", "[0, 1]", etc.
        Outputs:
            action: an action the agenet decides to take, in [0, 1, 2, 3].
        """

        # action is [0,1,2,3]: left, right, up, down
        state_list = json.loads(state)
        _state = np.array(state_list).reshape((1,2))
        q_values = self.model.predict(_state)[0]
        action_map = {0: [0, -1],  # left
                           1: [0, 1],  # right
                           2: [-1, 0],  # up
                           3: [1, 0],  # down
                           }
        chance = np.random.rand()
        if self.eps > chance:
            while True:
                action = np.random.randint(0,4)
                next = np.array(json.loads(state)) + np.array(action_map[action])
                if all(next>=0) and next[0]<=self.max_row and next[1]<=self.max_col:
                    break
        else:
            if state_list[0] == 0:
                q_values[2] -= 99
            if state_list[0] == self.max_row:
                q_values[3] -= 99
            if state_list[1] == 0:
                q_values[0] -= 99
            if state_list[1] == self.max_col:
                q_values[1] -= 99
            action = np.argmax(q_values)

        return action

    def learn(self, state, action, reward, next_state):
        """Learn from the environment.
        Args:
            state: str, the current state
            action: int, the action taken by itself
            reward: float, the reward after taking the action
            next_state: str, the next state

        Outputs:
            None
        """

        """Please Fill Your Code Here.
        """
        self.memory.append((np.array(json.loads(state)).reshape((1,2)), action, reward, np.array(json.loads(next_state)).reshape((1,2))))

        return 0


    def replay(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, self.batch_size)
        X = minibatch[0][0]
        Y = self.model.predict(X)
        for state, action, reward, next_state in minibatch:
            X = np.append(X, state, axis=0)
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target # action is a action_list index
            if state[0][0] == 0:
                target_f[0][2] = 0
            if state[0][0] == self.max_row:
                target_f[0][3] = 0
            if state[0][1] == 0:
                target_f[0][0] = 0
            if state[0][1] == self.max_col:
                target_f[0][1] = 0
            Y = np.append(Y, target_f, axis=0)
        self.model.fit(X[1:,],Y[1:,], batch_size=self.batch_size, epochs=20, verbose=False)
        print("the error value", self.model.evaluate(X[1:,],Y[1:,]))

if __name__ == "__main__":

    "define the parameters"
    agent_params = {
        "gamma" : 0.8,        # discounted rate
        "alpha" : 0.0001,        # learning rate
        "eps" : 0.9,          # initialize the e-greedy
        }

    max_iter = 20000

    # initialize the environment & the agent
    env = Environment()

    # the mouse cannot jump out of the grid
    agent_params["max_row"] = env.max_row - 1
    agent_params["max_col"] = env.max_col - 1
    smart_mouse = Agent(**agent_params)

    "start learning your policy"
    step = 0
    total_reward = 0
    for step in range(max_iter):
        current_state = env.reset()
        counter = 0
        while True:
            # act
            action = smart_mouse.do(current_state)

            # get reward from the environment
            next_state, reward, done = env.step(action, current_state)
            total_reward += reward
            # learn from the reward
            smart_mouse.learn(current_state, action, reward, next_state)
            current_state = next_state
            counter += 1
            if done:
                # print("Step {} done, reward {}".format(step, reward))
                break
            if counter>50:
                # print("fall into dead loop")
                break

        # epsilon decay
        if (step+1) % smart_mouse.update_model_freq == 0 and step>0:
            print("average reward in one step", total_reward/smart_mouse.update_model_freq)
            total_reward = 0
            smart_mouse.replay()
        if (step+1) % smart_mouse.update_target_freq == 0 and step>0:
            smart_mouse.update_target_network()

        if step % 100 == 0 and step > 0:
            eps = - 0.99 * step / max_iter + 1.0
            eps = np.maximum(0.1, eps)
            print("Eps:",eps)
            smart_mouse.eps = eps

    "evaluate your smart mouse"
    current_state = env.reset(False)
    trajectory_mat = env.reward.copy()
    trajectory_mat[0,0] = 5
    smart_mouse.eps = 0.0

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.ion()
    # smart_mouse.update_target_network()
    total_discounted_reward = 0
    epoch = 0
    while True:
        action = smart_mouse.do(current_state)
        next_state, reward, done = env.step(action, current_state)
        current_state = next_state
        trajectory_mat[json.loads(current_state)[0], json.loads(current_state)[1]] = 5

        ax.matshow(trajectory_mat, cmap="coolwarm")
        plt.pause(0.5)

        print("===> [Eval] state:{}, reward:{} <===".format(current_state, reward))
        print("state value: \n", smart_mouse.target_model.predict(np.array(json.loads(current_state)).reshape(1,2)))
        total_discounted_reward += reward * smart_mouse.gamma ** epoch
        epoch += 1
        if done:
            plt.title(datetime.datetime.now().ctime())
            plt.savefig("10_10_dqn.png")
            print("***" * 10)
            print("total discounted reward=", total_discounted_reward)
            print("***" * 10)
            break

