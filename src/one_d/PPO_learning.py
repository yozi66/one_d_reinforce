from stable_baselines3 import PPO
from OneDWorldEnv import OneDWorldEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np

env = OneDWorldEnv()
env = Monitor(env)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-2, n_steps=64)
model.learn(total_timesteps=500)

# --------------------------------
# Plot learning curve
# --------------------------------
results = env.get_episode_rewards()
plt.plot(results)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("PPO learning curve")
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
