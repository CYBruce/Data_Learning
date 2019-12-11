import matplotlib.pyplot as plt
import numpy as np
import pdb

np.random.seed(123456)

# set the reward matrix
reward = np.zeros((10,10))

# place the cheese
reward[9,9] = 4
reward[np.random.randint(0,10,10),np.random.randint(0,10,10)] = 1 

# place the trap
reward[np.random.randint(0,10,10),np.random.randint(0,10,10)] = -5 

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.matshow(reward, cmap="coolwarm")
# plt.show()

np.save("reward_10_10.npy", reward)



