# Reinforcement Learning

- The running example in this set of videos will be a helicopter flying. The input is the current state of the helicopter and the output should be the actions it take to correct itself
- Supervised learning doesn't work well here because its hard to find the exact right action given a particular state. So it's very hard to generate the right training set
- Instead we specify a reward function which incentivitizes good actions and trust the algorithm to teach itself the best way to perform the task based on the reward function
- Variables
    - $S$ is the state of 
    - $a$ action taken from this state
    - $R(s)$ is the reward from the state
    - $S'$ is the new state
    - $a'$ new action taken in $s'$
    - $Return$ = Sum of the rewards at each step = $R_1 + \gamma R_2 + \gamma^2 R_3$
    - $\gamma$ is the discount factor. Used to incentivize getting to good rewards quickly
    - $\pi$ is the policy that is used to describe what action to take based on the current state
- The return is what you use to determine which action to take. So we want to make sure that the policy tells us to take the right action to maximize the return
- Markov Decision Process (MDP) - formalism of reinforcement learning algorithm
    - Future only depends on the current state. Doesn't matter how you get there

## State action value function
- $Q^* = Q(s,a)$ = Return if you
    - Start in state $s$
    - Take action $a$ (once)
    - Behave optimally after that
- The best possible return from state $s$ is $max_a Q(s,a)$
- Bellman equation
    - $Q(s,a) = R(s) + \gamma * max_{a'}Q(s', a')$
    - Same as the intuitive description above
- Random (stochastic) environment
    - Random chance of going in a different state than you intended
    - Try to maximize the Expected Return to account for random environment
    - $Q(s,a) = R(s) + \gamma * E[max_{a'}Q(s', a')]$

## Continuous State Spaces
- The state space could be continuous, and most often is
    - Can also be a vector (indicating many different features)
- Lunar lander problem
    - Learn policy $\pi$ that give the state space s
        - $ s = [x, y, \dot{x}, \dot{y}, \theta, \dot{\theta}, l, r]$
        - Finds optimal action $a$ to take
- Train a neural network to compute $Q(s,a)$
    - Use neural network to compute Q(s, nothing), Q(s, left), Q(s, right), Q(s, main)
    - Input X to neural network is concat [s, a]
    - Output Y is Q(s,a)
- To get training data
    - Take a bunch of random actions from states, this gives you X
    - This also gives you the R(s) and the new $s'$. This is enough to compute Y since you have just a finite set of actions that you can take
- DQN (Deep Q Neural network)
    - Initialize neural network to randomly take a guess for weights to just guess $Q(s,a)$
    - Repeat
        - Take actions in the lunar lander, and store 10k pairs of $(s, a, R(s), s')$
        - Train neural network
            - Create new training set with 
                - X = [s a]
                - Y = $R(s) + \gamma * max_{a'} Q(s',a')$
            - Train $Q_{new}$ such that $Q_{new}(s,a) = y$
        - Set $Q = Q_{new}$
    - The intuition here is that as $Q$ neural network model will get better over time and you repeat these steps more and more since we have accurate estimates for the $R(s)$
    - Neural network architecture
        - X is [s a]
        - 2 hidden layers 
        - 1 final neuron in the output layer for Y
        - Optimized architecture
            - The problem with the proposed architecture is that you have to run inference 4 times
            - So lets make the output layer 4 neurons, one for each action
                - Question: But how does this work since you don't specify actions to each layer? It could say assign each of the 4 neurons the left action
                   - You don't! It's just an estimate (like the whole neural network)
    - Choosing actions while still learning (Epsilon greedy policy)
        - With some probability $1 - \epsilon$ pick the action $a$ that maximized $Q(s,a)$, otherwise pick a random action
            - Helps avoid local maxima
    - Additional optimizations
        - Mini-batches
            - Useful when the training set is very very large
            - Main idea: Instead of using all the training set, for each iteration of gradient descent use $m'$ as a subset of the training set $m$ so each iteration of gradient descent can run quicker
            - Will tend towards the minimum (but not as reliably as regular gradient descent)
        - Soft updates
            - $Q = Q_{new}$ might be too abrupt of a change and maybe a new one
            - Main idea, use the weights from $Q_{new}$ but multiply it by a small number so you can get it to slowly move towards $Q_{new}$
                - $w = 0.01*w_{new} + 0.99*w$
            
 