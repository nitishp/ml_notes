# Reinforcement Learning

- The running example in this set of videos will be a helicopter flying. The input is the current state of the helicopter and the output should be the actions it take to correct itself
- Supervised learning doesn't work well here because its hard to find the exact right action given a particular state. So it's very hard to generate the right training set
- Instead we specify a reward function which incentivitizes good actions and trust the algorithm to teach itself the best way to perform the task based on the reward function
- Variables
    - $S$ is the state of 
    - $a$ action taken from this state
    - $R(s)$ is the reward from the state
    - $S'$ is the new state
    - $Return$ = Sum of the rewards at each step = $R_1 + \gamma R_2 + \gamma^2 R_3$
    - $\gamma$ is the discount factor. Used to incentivize getting to good rewards quickly
    - $\pi$ is the policy that is used to describe what action to take based on the current state
- The return is what you use to determine which action to take. So we want to make sure that the policy tells us to take the right action to maximize the return
- Markov Decision Process (MDP) - formalism of reinforcement learning algorithm
    - Future only depends on the current state. Doesn't matter how you get there
