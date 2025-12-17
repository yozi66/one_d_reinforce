from OneDWorldEnv import OneDWorldEnv

env = OneDWorldEnv(length=5, max_steps=10)
obs, info = env.reset()
print("Initial obs:", obs)
env.render()
for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}, action={action}, obs={obs}, reward={reward}, done={terminated or truncated}")
    env.render()
    if terminated or truncated:
        break
