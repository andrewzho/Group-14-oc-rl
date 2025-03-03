import numpy as np
import matplotlib.pyplot as plt
rewards = np.load("rewards.npy")
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Rewards")
plt.savefig("docs/rewards.png")