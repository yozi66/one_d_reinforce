import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor  
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env = Monitor(env)
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# visualization of learning curve

results = env.get_episode_rewards()
plt.plot(results)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("DQN learning curve")
plt.show()

# render

render_env = gym.make("CartPole-v1", render_mode="human")
obs, info = render_env.reset()

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = render_env.step(action)

    if terminated or truncated:
        obs, info = render_env.reset()

render_env.close()


