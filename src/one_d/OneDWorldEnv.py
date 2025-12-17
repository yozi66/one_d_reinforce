import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OneDWorldEnv(gym.Env):
    """
    A simple 1D line:
    positions: 0 ... length-1
    start at 0, goal at length-1
    actions: 0 = left, 1 = right
    reward: +1 when reaching the goal, small -0.01 otherwise
    episode ends at goal or after max_steps
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, length=5, max_steps=20):
        super().__init__()
        self.length = length
        self.max_steps = max_steps

        # State is a single integer position
        self.observation_space = spaces.Discrete(self.length)
        # 0 = left, 1 = right
        self.action_space = spaces.Discrete(2)

        self.state = None
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # start on the left
        self.steps_taken = 0
        info = {}
        return np.array(self.state, dtype=np.int32), info

    def step(self, action):
        self.steps_taken += 1

        # Apply action
        if action == 0 and self.state > 0:  # left
            self.state -= 1
        elif action == 1 and self.state < self.length - 1:  # right
            self.state += 1

        # Check termination
        terminated = self.state == self.length - 1  # reached goal
        truncated = self.steps_taken >= self.max_steps

        # Reward
        reward = 1.0 if terminated else -0.01

        info = {}
        return np.array(self.state, dtype=np.int32), reward, terminated, truncated, info

    def render(self):
        line = ["-"] * self.length
        line[self.state] = "A"
        print("".join(line))
