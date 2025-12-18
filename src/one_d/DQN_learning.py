import pandas as pd
from stable_baselines3 import DQN
from OneDWorldEnv import OneDWorldEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np

env = OneDWorldEnv()
env = Monitor(env)
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-1,
            buffer_size=500,
            learning_starts=100,
            train_freq=1,
            # EXPLORATION TUNING
            exploration_initial_eps=0.5,
            exploration_final_eps=0.05,
            exploration_fraction=0.5,
)
model.learn(total_timesteps=1_000)

# --------------------------------
# Plot learning curve
# --------------------------------
results = env.get_episode_rewards()
plt.plot(results)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("DQN learning curve")
plt.show()

# --------------------------------
# Use the learned policy
# --------------------------------
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    env.render()
    done = terminated or truncated
