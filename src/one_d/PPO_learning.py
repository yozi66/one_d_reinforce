from stable_baselines3 import PPO
from OneDWorldEnv import OneDWorldEnv

env = OneDWorldEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=10_000)

obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    env.render()
    done = terminated or truncated
