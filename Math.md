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

The Q-table is a matrix directly storing $Q(s,a)$. The [Q_learning](src/one_d/Q_learning.py) algorithm starts with zeroes and learns the $Q$ function by experience. 

After each action executed, $Q(s,a)$ is adjusted using the actual reward $r$ for action $a$ and $Q(s',a')$ as learned from the previous episodes. 

The algorithm usually chooses the (believed) optimal action, but in some cases it "explores" the state space and other possible actions with a random action.

DQN learning
------------

DQN stands for Deep Q-Network. For non-trivial problems, the $Q$ function cannot be stored as a matrix because it does not fit in memory. We use a deep neural network to learn the $Q$ function. The state (weights) of the neural network is $\theta$.

At training step $t$:

1️⃣ Online network (current, being trained): 
```math
Q_{\theta}(s, a)
```

2️⃣ Target network (frozen snapshot):
```math
Q_{\theta^{\boldsymbol{–}}}(s, a)
```

3️⃣ Target value (Bellman target):
```math
y=r+\gamma \max_{⁡a'}Q_{\theta^{\boldsymbol{–}}}(s', a')
```

4️⃣ Loss:
```math
\mathcal{L}(\theta)=(Q_{\theta}(s, a)−y)^2
```

5️⃣ Update the weights (simple gradient descent):
```math
\theta_{new} = \theta_{old} - \alpha \nabla_{\theta} \mathcal{L}(\theta)
```
where $\alpha$ is the learning factor and $\nabla \mathcal{L}$ is the gradient of the loss function. 
 
By using a separate Target Network ($\theta^{\boldsymbol{–}}$, which is just an older copy of the weights), the target stays "frozen" for a few hundred steps. This gives the main network a stable goal to aim for.

DQN doesn't update on every single step; it updates using Mini-Batches from the Experience Replay buffer.

1. **Sample:** Take a random batch of experiences $(s,a,r,s')$ from the buffer.

2. **Forward Pass (Prediction):** Pass the states ($s$) through the current network to get the predicted $Q(s,a)$.

3. **Forward Pass (Target):** Pass the next states ($s'$) through the Target Network to find the best possible future value ($\max Q$).

4. **Compute Loss:** Calculate the difference between the Bellman Target and the Prediction.

5. **Backpropagation:** Calculate the gradient of the loss with respect to the weights ($\theta$).

6. **Optimizer Step:** An optimizer (like Adam) nudges the weights $\theta$ in the direction that reduces the loss.

**Adam** (short for Adaptive Moment Estimation) is the most popular optimizer in Deep Learning today. Adam improves on this with two key "superpowers": **Momentum** and **Adaptive Learning Rates**.

In Simple Gradient Descent (SGD), you use one fixed learning rate ($\alpha$) for every single weight in your neural network. 

A. **Momentum**

In RL, gradients can be very noisy (one bad game might give a "weird" gradient).

* **SGD:** Reacts instantly to every noisy gradient, causing the weights to zig-zag wildly.

* **Adam:** Keeps a "moving average" of previous gradients. It builds up speed in the right direction and ignores the tiny, random zig-zags. This is known as the **First Moment**.

B. **Adaptive Learning Rates**

Not all weights are equal. Some weights in your network might need to change a lot, while others barely need to move.

* **SGD:** Moves all weights by the same scale.
* **Adam:** Tracks how much each weight has been changing (the **Second Moment** or variance).
  * If a weight is getting huge, frequent updates, Adam slows it down.
  * If a weight is barely moving, Adam speeds it up.

PPO learning
------------

Proximal Policy Optimization (PPO) is an **actor-critic** algorithm.
* **Actor** -> decides what action to take
* **Critic** -> estimates how good the state is

The policy (the brain of the Actor) is a neural network (with $\theta$ weights) that gives the probability of action $a$ in state $s$. It is analogous to the Q-table or the Q-network in DQN:

```math
\pi_{\theta}( a | s)
```

The value (the brain of the Critic) is a neural network (with $\theta$ weights) that gives the value of a state:

```math
V_{\theta}(s)
```
