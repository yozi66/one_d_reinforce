import random
import matplotlib.pyplot as plt
import numpy as np

from OneDWorldEnv import OneDWorldEnv

env = OneDWorldEnv(length=50, max_steps=100)

n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table: states x actions
Q = np.zeros((n_states, n_actions))

alpha = 0.1      # learning rate
gamma = 0.99     # discount factor
epsilon = 0.2    # exploration rate
n_episodes = 500

episode_returns = []

for episode in range(n_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # ε-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next = np.max(Q[next_state])
        td_target = reward + gamma * best_next * (0 if done else 1)
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)

# Plot learning curve
plt.plot(episode_returns)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Q-learning in 1D World")
plt.show()
