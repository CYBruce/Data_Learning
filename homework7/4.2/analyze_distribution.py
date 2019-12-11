# -*- coding: utf-8 -*-

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20191109"


import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import pdb
import datetime

np.random.seed(2019)

"define the environment"
class Environment:
    def __init__(self):
        # set the reward matrix
        reward = np.zeros((4,4))

        # place the cheese
        reward[3,3] = 4
        reward[3,0] = 2
        reward[[0,0,2],[3,1,1]] = 1

        # place the poison
        reward[[1,1,3],[3,1,1]] = -5

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
                init_state = np.random.randint(0,4,(2))
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

        # self.Q = defaultdict(lambda: np.array([.0, .0, .0, .0]))
        self.Q = defaultdict(lambda: np.random.rand(4))


    def do(self, state):
        """Know the current state, choose an action.
        Args:
            state: str, as "[0, 0]", "[0, 1]", etc.
        Outputs:
            action: an action the agenet decides to take, in [0, 1, 2, 3].
        """

        # action is [0,1,2,3]: left, right, up, down
        _state = json.loads(state)
        if _state[0] == 0:
            self.Q[state][2] = 0
        if _state[0] == self.max_row:
            self.Q[state][3] = 0
        if _state[1] == 0:
            self.Q[state][0] = 0
        if _state[1] == self.max_col:
            self.Q[state][1] = 0
        action_map = {0: [0, -1],  # left
                           1: [0, 1],  # right
                           2: [-1, 0],  # up
                           3: [1, 0],  # down
                           }
        chance = np.random.rand()
        if self.eps > chance:
            while True:
                action = np.random.randint(0,4)
                next = np.array(_state) + np.array(action_map[action])
                if all(next>=0) and next[0]<=self.max_row and next[1]<=self.max_col:
                    break
        else:
            action = np.argmax(self.Q[state])
        #print(_state,action,self.Q[state])

        """Please Fill Your Code Here.
        """
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
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])

        return 0


if __name__ == "__main__":

    "define the parameters"
    agent_params = {
        "gamma" : 0.8,        # discounted rate
        "alpha" : 0.1,        # learning rate
        "eps" : 0.9,          # initialize the e-greedy
        }

    max_iter = 10000

    # initialize the environment & the agent
    env = Environment()

    # the mouse cannot jump out of the grid
    agent_params["max_row"] = env.max_row - 1
    agent_params["max_col"] = env.max_col - 1
    smart_mouse = Agent(**agent_params)

    "start learning your policy"
    step = 0
    for step in range(max_iter):
        current_state = env.reset()
        while True:
            # act
            action = smart_mouse.do(current_state)

            # get reward from the environment
            next_state, reward, done = env.step(action, current_state)

            # learn from the reward
            smart_mouse.learn(current_state, action, reward, next_state)

            current_state = next_state

            if done:
                print("Step {} done, reward {}".format(step, reward))
                break

        # epsilon decay
        if step % 100 == 0 and step > 0:
            eps = - 0.99 * step / max_iter + 1.0
            eps = np.maximum(0.1, eps)
            print("Eps:",eps)
            smart_mouse.eps = eps

    # "evaluate your smart mouse"
    # current_state = env.reset(False)

    # smart_mouse.eps = 0.0
    reward = env.raw_reward
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    plt.ion()
    epoch = 0

    reward_learned = np.zeros((4, 4))
    # state: str, as "[0, 0]", "[0, 1]"
    q_table = smart_mouse.Q.copy()
    for state in q_table.keys():
        _reward = 0
        count = 0
        state_list = json.loads(state)
        if state_list[0]>0:
            state_above = "[{}, {}]".format(state_list[0]-1,state_list[1])
            _reward += (smart_mouse.Q[state_above][3] - smart_mouse.gamma * np.max(smart_mouse.Q[state]))
            count+=1
        if state_list[0]<smart_mouse.max_row:
            state_above = "[{}, {}]".format(state_list[0]+1,state_list[1])
            _reward += (smart_mouse.Q[state_above][2] - smart_mouse.gamma * np.max(smart_mouse.Q[state]))
            count+=1
        if state_list[1]>0:
            state_above = "[{}, {}]".format(state_list[0],state_list[1]-1)
            _reward += (smart_mouse.Q[state_above][1] - smart_mouse.gamma * np.max(smart_mouse.Q[state]))
            count+=1
        if state_list[1]<smart_mouse.max_col:
            state_above = "[{}, {}]".format(state_list[0],state_list[1]+1)
            _reward += (smart_mouse.Q[state_above][0] - smart_mouse.gamma * np.max(smart_mouse.Q[state]))
            count+=1
        reward_learned[state_list[0]][state_list[1]] = _reward / count

    ax.matshow(reward_learned, cmap="coolwarm")
    plt.title("the learned reward_matrix")
    bx = fig.add_subplot(1, 2, 2)
    bx.matshow(reward, cmap="coolwarm")
    plt.title("the real reward_matrix")
    plt.savefig("distribution.png")
    print("the learned reward matrix")
    print(reward_learned)
    print("the real reward matrix")
    print(reward)
    print("mean squared error = ",np.sum(np.square(np.abs(np.array(reward_learned)-np.array(reward))))/16)
    print(np.square(np.abs(np.array(reward_learned)-np.array(reward))), np.argmax(np.square(np.abs(np.array(reward_learned)-np.array(reward)))))
    print(smart_mouse.Q)
'''
It's not hard to find the most different part is (3, 0) where the reward is 2, but the learned reward is 1.2
how to explain, because this part is rarely visited as the elipson decrease, causing some bias
'''
