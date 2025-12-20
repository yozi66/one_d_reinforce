Reinforcement Learning
======================

Some formulas for reinforcement learning. 

Bellman optimality equation
---------------------------

$Q(s,a)$ is the value of action $a$ in state $s$.

```math
Q(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q(s', a')\right]
```

According to the Bellman equation, the optimal value of the Q function is equal to the expected value $\mathbb{E}$ of the sum of the reward $r$ received for action $a$ in state $s$ and the discounted value of the optimal action $a'$ in the next state $s'$. The discount factor is $\gamma$.

Q-table
-------

TBD.
