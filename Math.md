Reinforcement Learning
======================

Some formulas and algorithms for reinforcement learning. 

Bellman optimality equation
---------------------------

$Q(s,a)$ is the value of action $a$ in state $s$.

```math
Q(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q(s', a')\right]
```

According to the Bellman equation, the optimal value of the Q function is equal to the expected value $\mathbb{E}$ of the sum of the reward $r$ received for action $a$ in state $s$ and the discounted value of the optimal action $a'$ in the next state $s'$. The discount factor is $\gamma$.

Q-table algorithm
-----------------

The Q-table is a matrix directly storing Q(s,a). The [Q_learning](src/one_d/Q_learning.py) algorithm starts with zeroes and learns the $Q$ function by experience. 

After each action executed, $Q(s,a)$ is adjusted using the actual reward $r$ for action $a$ and $Q(s',a')$ as learned from the previous episodes. 

The algorithm usually chooses the (believed) optimal action, but in some cases it "explores" the state space and other possible actions with a random action.
