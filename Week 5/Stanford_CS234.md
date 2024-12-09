# Stanford CS234: Reinforcement Learning | Winter 2019

[](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

Reinforced Learning → How can an intelligent agent make good sequence of decisions 

## Key aspects of Reinforced learning

- Optimization
    - Goal is to find an optimal way to make decision (i.e. That **yields** best outcome) or at least a very good strategy.
- Delayed Consequence
    - Decision impact things much later;
    - Because we don’t receive immediate outcome feedback → it’s hard to do the **credit assignment problem** → which is how we figure out the cause of relationship of things you did in the past, and  the outcomes in the future.
- Exploration
    - Learning about the world by making decision
    - Censored Data → You only know the outcome once you have made that decision.
    - Decision impact what we learn
    - Policies are mapping of decision based on experience  & why not just pre program the policies ?
- Generalization
    - We need to learn from data directly & have a higher level representation of task;
    - So that if we run into a new situation, we have a higher level representation of task that we want to achieve.

---

### AI Planning (vs RL)

> It involves all the components of RL, except exportation.
> 
> - Because we already know how the environment works & all the rules associated with it.

### Supervised ML (vs ML)

> It involves all the components of RL, except exportation & Delayed consequence.
> 
> - Because we already know how the environment works & all the rules associated with it.
> - Because it already has the ‘experience’ already, because the labels are already provided to you.

### Unsupervised ML (vs ML)

> It involves all the components of RL, except exportation & Delayed consequence.
> 
> - Because we already know how the environment works & all the rules associated with it.
> - Because it already has the ‘experience’ already, because the labels are not provided to you.

### Imitation ML (vs ML)

> It involves all the components of RL, except exportation.
> 
> - Because it learns from experience — of others
> - it first observes (a human / Artificial decision making process); than makes decision based on the outcome it observed.
- Challenges of IL →  https://youtu.be/FgzM3zpZ55o?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&t=876

---

## Intro to Sequential Decision Making under uncertainty.

> We think of an **agent** who interacts with the **world** via **actions** and the world outputs **observations** and **rewards**. Our goal is to maximize total expected future reward.

- Goal: Select actions to maximize total **expected** future reward
    - May require balancing immediate & long term rewards
    - May require strategic behavior to achieve high rewards

**Reward Hacking** → When the agent learns exactly what the reward function is,
So, if the agent is set up in such a way that it gets +1 for correct answers & -1 for wring answers; and there are 2 types of question, one easier than other; then the agent will only provide easy question to increase the likelihood of it receiving the reward.  

[CS234: Reinforcement Learning Spring 2024](https://web.stanford.edu/class/cs234/modules.html)

---

## Key Concepts in Reinforcement Learning

### Markov Decision Processes (MDPs)

- MDPs model the environment as a set of states, actions, and transition probabilities.
- The agent's state is a function of the history of observations and actions.
- The Markov assumption states that the future is independent of the past given the present state.

### Bandits

- Bandits are a simplified version of MDPs where actions do not affect the next state.

### Deterministic vs. Stochastic Dynamics

- In deterministic environments, actions lead to a single next state.
- In stochastic environments, actions lead to a distribution over possible next states.

### Reward Models and Value Functions

- The reward model specifies the immediate reward for taking an action in a state.
- The value function represents the expected discounted sum of future rewards under a given policy.

### Model-based vs. Model-free Approaches

- Model-based agents maintain explicit representations of the transition and reward models.
- Model-free agents learn value functions and policies directly from experience, without an explicit model.

## Common Reinforcement Learning Challenges

### Planning

- Even with a known model, an agent must compute the best actions to take to maximize long-term reward.

### Exploration vs. Exploitation

- Agents must balance exploring the environment to gather information and exploiting their current knowledge to maximize reward.

### Credit Assignment

- Agents must determine which past actions were responsible for current rewards or penalties, a challenging problem due to delayed consequences.

### Generalization

- Agents must be able to generalize from their experiences to handle novel situations, rather than relying on a lookup table of all possible states and actions.

| Concept | Description |
| --- | --- |
| Markov Decision Process (MDP) | A framework for modeling sequential decision-making under uncertainty, consisting of states, actions, transition probabilities, and rewards. |
| Bandit | A simplified MDP where actions do not affect the next state. |
| Deterministic Dynamics | Environments where actions lead to a single next state. |
| Stochastic Dynamics | Environments where actions lead to a distribution over possible next states. |
| Reward Model | Specifies the immediate reward for taking an action in a state. |
| Value Function | Represents the expected discounted sum of future rewards under a given policy. |
| Model-based Approach | Agents maintain explicit representations of the transition and reward models. |
| Model-free Approach | Agents learn value functions and policies directly from experience, without an explicit model. |
| Exploration vs. Exploitation | The challenge of balancing gathering information about the environment and maximizing reward based on current knowledge. |
| Credit Assignment | The problem of determining which past actions were responsible for current rewards or penalties. |
| Generalization | The ability to apply learned knowledge to novel situations, rather than relying on a lookup table. |

# Reinforcement Learning: Exploration vs. Exploitation

## Description

This set of notes covers the key concepts of reinforcement learning, including the exploration-exploitation dilemma, policy evaluation, and policy control. It also discusses the differences between finite and infinite horizon problems, and how that affects the optimal decision-making strategy.

## Key Terms and Important Points

**Exploration vs. Exploitation**

- Exploration involves trying new actions or states that may lead to better long-term rewards, even if they seem worse in the short-term.
- Exploitation involves taking actions that are known to yield high rewards based on past experience.
- The agent must balance exploration and exploitation to maximize long-term rewards.

**Finite vs. Infinite Horizon Problems**

- In a finite horizon problem, the agent only has a limited number of decisions to make (e.g., 5 days in a restaurant).
- In an infinite horizon problem, the agent can repeat the decision-making process indefinitely.
- The optimal policy for a finite horizon problem is often non-stationary, meaning the decision depends on the current time step. In an infinite horizon problem, the optimal policy is typically stationary.

**Indefinite Horizon Problems**

- These are problems where the agent doesn't know when the finite horizon will end.
- One way to model this is as an infinite horizon problem with termination states.

**Policy Evaluation vs. Policy Control**

- Policy evaluation involves assessing the quality of a given policy, without trying to improve it.
- Policy control involves finding the optimal policy, which typically requires policy evaluation as a sub-component.

**Counterfactual Reasoning**

- Reinforcement learning allows for counterfactual reasoning, where data from previous policies can be used to evaluate the performance of new policies without having to try them all out.

**Learning Reward Functions and Policies Simultaneously**

- It is possible to learn both the reward function and the optimal policy at the same time, by using the prior experience to inform the next policy or set of policies to try.

| Concept | Description |
| --- | --- |
| Exploration | Trying new actions or states that may lead to better long-term rewards |
| Exploitation | Taking actions that are known to yield high rewards based on past experience |
| Finite Horizon | The agent only has a limited number of decisions to make |
| Infinite Horizon | The agent can repeat the decision-making process indefinitely |
| Policy Evaluation | Assessing the quality of a given policy |
| Policy Control | Finding the optimal policy |
| Counterfactual Reasoning | Using data from previous policies to evaluate new policies |

# Reinforcement Learning: Markov Decision Processes

## Overview of Reinforcement Learning Concepts

- Reinforcement learning involves a model of value and a policy
- The policy is a mapping from the state to the action the agent should take
- The policy can be good or bad, and is evaluated based on the expected discounted sum of rewards

## Model, Policy, and Value Function

- **Model**: A representation of the world and how it changes in response to actions
    - Can be stochastic or deterministic
    - Includes a reward model that specifies the expected reward for taking an action in a state
- **Policy**: A function that maps agent states to actions
- **Value Function**: The expected discounted sum of rewards from being in a state and following a particular policy

## Markov Processes

- **Markov Process**: A stochastic process where the future is independent of the past, given the present state
- **Markov Chain**: A sequence of random states where the transition dynamics satisfy the Markov property
- The transition dynamics can be represented as a matrix, where each element specifies the probability of transitioning from one state to another
- Markov processes have no actions or rewards, they just describe the evolution of the state over time

## Markov Reward Processes

- **Markov Reward Process (MRP)**: A Markov process with a reward function
- The reward function specifies the expected reward for being in a particular state
- The **return** is the discounted sum of rewards over an episode
- The **value function** is the expected return from a state
- Methods for computing the value function:
    1. Simulation: Sample episodes and average the returns 
    2. Analytical solution: Use matrix inversion 
    3. Dynamic programming: Iterative Bellman backup 

## Markov Decision Processes

- **Markov Decision Process (MDP)**: An MRP with actions
- The dynamics model now specifies the probability distribution over next states given the current state and action
- The reward function can depend on the state, action, and next state
- **Policy**: A mapping from states to actions
    - Can be deterministic or stochastic
- The optimal policy is deterministic and stationary
- Computing the optimal policy:
    1. Policy search: Enumerate all possible policies and evaluate them 
    2. Policy iteration: Start with a policy, evaluate it, and iteratively improve it 
        - Involves computing the state-action value function

### State-Action Value Function

- The state-action value function, Q(s, a), represents the expected return if you take action a in state s and then follow the current policy.
- This allows us to improve the policy by taking the action that maximizes Q(s, a) for each state.

| Key Term | Definition |
| --- | --- |
| Model | A representation of the world and how it changes in response to actions |
| Policy | A mapping from agent states to actions |
| Value Function | The expected discounted sum of rewards from being in a state and following a particular policy |
| Markov Process | A stochastic process where the future is independent of the past, given the present state |
| Markov Chain | A sequence of random states where the transition dynamics satisfy the Markov property |
| Markov Reward Process | A Markov process with a reward function |
| Return | The discounted sum of rewards over an episode |
| Markov Decision Process | An MRP with actions |
| State-Action Value Function | The expected return if you take action a in state s and then follow the current policy |

# Policy Iteration and Value Iteration

Policy iteration is an algorithm for finding the optimal policy in a Markov Decision Process (MDP) when the dynamics and reward models are known. It involves iteratively evaluating and improving the current policy until convergence.

**Key Points:**

- Policy evaluation computes the value of the current policy.
- Policy improvement computes a new policy that is at least as good as the current policy.
- The new policy is guaranteed to be monotonically better than the old policy.
- Once the policy stops changing, it will never change again.
- There is a maximum number of policy iterations equal to the number of possible policies (a^s).

## Value Iteration

**Description:**Value iteration is an alternative approach to finding the optimal policy in an MDP. Instead of maintaining a policy and iteratively improving it, value iteration computes the optimal value function directly by iterating the Bellman backup operator.

**Key Points:**

- The Bellman equation defines the relationship between the value of a state and the values of its successor states.
- The Bellman backup operator applies this equation to transform an old value function into a new one.
- Value iteration initializes the value function to 0 and repeatedly applies the Bellman backup until convergence.
- The Bellman backup is a contraction operator, meaning it shrinks the distance between value functions.
- This contraction property guarantees that value iteration will converge to a unique fixed point.
- The initialization of the value function does not affect the final converged solution.

**Table: Comparison of Policy Iteration and Value Iteration**

| Property | Policy Iteration | Value Iteration |
| --- | --- | --- |
| Maintains | Policy | Value Function |
| Iterative Process | Evaluate policy, improve policy | Apply Bellman backup |
| Convergence | Monotonic improvement | Contraction mapping |
| Maximum Iterations | a^s | Depends on convergence criteria |
| Initialization | Arbitrary policy | Arbitrary value function |

## Conclusion

Both policy iteration and value iteration are powerful algorithms for solving MDPs when the dynamics and reward models are known. Policy iteration maintains a policy and iteratively improves it, while value iteration computes the optimal value function directly. The choice between the two approaches depends on the specific problem and the available computational resources.

---

# Model-Free Policy Evaluation

## Description

This section covers the concept of model-free policy evaluation, which involves estimating the value of a policy without having access to the underlying model of the environment. It discusses three main approaches: dynamic programming, Monte Carlo policy evaluation, and temporal difference (TD) learning.

## Key Points

- **Dynamic Programming**: Requires knowledge of the transition and reward models to compute the value of a policy through iterative updates.
- **Monte Carlo Policy Evaluation**: Estimates the value of a policy by averaging the returns from multiple episodes, without needing a model.
- **Temporal Difference (TD) Learning**: Combines aspects of dynamic programming and Monte Carlo methods, using bootstrapping to update value estimates after each transition.

## Dynamic Programming for Policy Evaluation

- Dynamic programming can be used to evaluate a policy when the transition and reward models are known.
- The value function is initialized to 0 and iteratively updated until convergence.
- The value of a state is the expected immediate reward plus the discounted expected future value.
- Convergence is determined by comparing the difference in value functions between iterations.

## Monte Carlo Policy Evaluation

- Monte Carlo policy evaluation estimates the value of a policy by averaging the returns from multiple episodes.
- The first-visit Monte Carlo method updates the value of a state only on the first visit within an episode.
- The every-visit Monte Carlo method updates the value of a state every time it is visited within an episode.
- Monte Carlo methods are unbiased but can have high variance, especially in the early stages.
- Monte Carlo methods require episodic tasks, where the process terminates and a full return can be observed.

## Temporal Difference (TD) Learning

- TD learning combines aspects of dynamic programming and Monte Carlo methods.
- Instead of waiting for the full return, TD learning updates the value of a state using the immediate reward and the estimated value of the next state.
- The TD error compares the estimated value of the current state to the sum of the immediate reward and the discounted estimated value of the next state.
- TD learning is a biased estimator but can have lower variance than Monte Carlo methods.
- TD learning can be applied in both episodic and continuous tasks, as it does not require waiting for the full return.

## Comparison of Estimators

- Bias and variance are important considerations when comparing different policy evaluation methods.
- Unbiased estimators, like first-visit Monte Carlo, converge to the true value but may have high variance.
- Biased estimators, like TD learning, can have lower variance but may not converge to the true value.
- The choice of method depends on the specific problem and the trade-offs between bias and variance.

## Example: Mars Rover Domain

- A simple example is used to illustrate the differences between first-visit Monte Carlo, every-visit Monte Carlo, and TD learning.
- The first-visit Monte Carlo method updates the value of each state only on the first visit, resulting in a vector of [1, 1, 1, 0, 0, 0, 1].
- The every-visit Monte Carlo method updates the value of each state every time it is visited, resulting in a vector of [1, 1, 1, 1, 1, 1, 1].
- The TD learning method updates the value of each state immediately after the transition, resulting in a vector of [0, 0, 0, 0, 0, 0, 1].

| Method | Value Estimates |
| --- | --- |
| First-visit Monte Carlo | [1, 1, 1, 0, 0, 0, 1] |
| Every-visit Monte Carlo | [1, 1, 1, 1, 1, 1, 1] |
| TD Learning | [0, 0, 0, 0, 0, 0, 1] |

---

# Policy Evaluation Algorithms: Strengths and Weaknesses

## Description

This set of notes explores the key properties and trade-offs of different policy evaluation algorithms, including Dynamic Programming (DP), Monte Carlo (MC), and Temporal Difference (TD) learning. We'll examine how these methods handle factors like model-free learning, non-episodic domains, Markovian assumptions, convergence guarantees, and unbiased value estimates.

## Key Terms and Concepts

### Trajectory and Tuple Updates

- Policy evaluation algorithms update their value estimates by iterating through a trajectory of states, actions, and rewards (S, A, R, S') represented as tuples. (00:53:44 - 00:54:05)
- The order in which these tuples are received can impact the convergence and accuracy of the value estimates.

### Convergence and Unbiased Estimates

- DP, MC, and TD learning are all guaranteed to converge to the true value function in the limit, under certain assumptions.
- However, only MC provides an unbiased estimate of the value, while TD and DP can be biased for finite data.

### Model-Free vs. Model-Based

- DP requires a model of the environment, while MC and TD are model-free and can learn directly from samples.
- This makes MC and TD more flexible for unknown or non-Markovian domains, but DP can be more data-efficient if a model is available.

### Batch vs. Online Learning

- In the batch setting, where all data is available upfront, MC and TD converge to the same value estimates, but for different reasons.
- MC minimizes mean-squared error, while TD learns the maximum likelihood MDP model and performs dynamic programming.

## Comparison of Algorithms

| Property | Dynamic Programming | Monte Carlo | Temporal Difference |
| --- | --- | --- | --- |
| Usable without model | No | Yes | Yes |
| Handles continuing domains | Yes | No | No |
| Requires Markovian assumption | Yes | No | Yes |
| Converges to true value (limit) | Yes | Yes | Yes |
| Provides unbiased estimate | No | Yes | No |

# Model-Free Control: Exploring and Optimizing Without a Model

This section covers how an agent can make good decisions when it doesn't know how the world works, but still wants to maximize its expected discounted sum of rewards. We'll discuss methods for model-free control, including Monte Carlo control and temporal difference (TD) methods like Sarsa.

## Key Points

- **Sessions**: Optional sessions are being offered to go deeper into the course material and discuss homework. Attendance will earn 1% extra credit.
- **Model-Free Control**: Focuses on how an agent can make good decisions without an explicit model of the environment's dynamics and rewards.
- **Exploration vs. Exploitation**: The agent must balance exploring unknown actions to gather more information, and exploiting its current knowledge to maximize rewards.
- **Monte Carlo Control**: Estimates the state-action value function (Q-function) directly from experience, then uses it to improve the policy.
- **Temporal Difference (TD) Methods**: Update the Q-function after each transition, bootstrapping from the next state's estimated value, rather than waiting for the end of an episode.
- **Sarsa**: A TD method that updates the Q-function using the next state-action pair actually taken, rather than the greedy maximum.

## Monte Carlo Control

- Initialize a random policy π₀
- Repeat:
    - Generate an episode following π_i using an ε-greedy policy
    - For each state-action pair (s, a) visited:
        - Increment the visit count N(s, a)
        - Accumulate the total discounted return G
        - Q(s, a) ← Q(s, a) + 1/N(s, a) * (G - Q(s, a))
    - π_i+1 ← greedy w.r.t. Q

## Temporal Difference (TD) Methods

- Initialize a random ε-greedy policy
- Repeat:
    - Take action a, observe r, s'
    - Take next action a'
    - Q(s, a) ← Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a))
    - s ← s', a ← a'

## Sarsa

- Initialize a random ε-greedy policy
- Repeat:
    - Take action a, observe r, s'
    - Take next action a'
    - Q(s, a) ← Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a))
    - s ← s', a ← a'

## Comparison of Methods

| Method | Updates Q | Explores |
| --- | --- | --- |
| Monte Carlo | After episode | ε-greedy |
| Sarsa | After each transition | ε-greedy |
| Q-Learning | After each transition | Greedy w.r.t. Q |

# Reinforcement Learning: Sarsa, Q-Learning, and Maximization Bias

## Sarsa vs. Q-Learning

**Description:** Sarsa and Q-Learning are two different approaches to reinforcement learning, each with its own advantages and disadvantages.

**Key Points:**

- Sarsa is more "realistic" in the early stages, as it considers the actual next action that will be taken, rather than just the maximum possible future reward (as Q-Learning does). This can be beneficial in domains with a lot of negative rewards early on.
- However, empirically, Q-Learning often performs better overall, as the optimistic approach can aid exploration.
- Both Sarsa and Q-Learning will converge to the same optimal policy, given the right conditions.

## Convergence Conditions

**Description:** For both Sarsa and Q-Learning to converge to the optimal policy, certain conditions must be met.

**Key Points:**

- The learning rate (α) must be properly set:
    - α = 1 will not allow convergence, as no information from the past is remembered.
    - α = 0 means no updates are made, also preventing convergence.
    - Empirically, small or slowly decaying constants are often used for α.
- The exploration policy must satisfy the "Greedy in the Limit with Infinite Exploration" (GLIE) condition:
    - The policy must become more greedy over time, but still explore all state-action pairs infinitely often.
    - This can be difficult to achieve in some domains, such as when certain states become unreachable.

## Q-Learning Update

**Description:** The Q-Learning update rule is similar to Sarsa, but with a key difference.

**Key Points:**

- Q-Learning updates the Q-value for the current state-action pair, using the maximum expected future reward, rather than the actual next action taken.
- This means Q-Learning can update the Q-value earlier, as it doesn't need to wait for the next action to be taken.
- Q-Learning only needs to update the policy for the current state, rather than potentially updating multiple states. This can be computationally helpful.

## Initialization and Maximization Bias

**Description:** The initialization of the Q-function and the maximization step in Q-Learning can introduce bias.

**Key Points:**

- Initializing the Q-function optimistically can be helpful empirically, even though it doesn't affect the asymptotic convergence.
- The "maximization bias" can occur when taking the max over estimated Q-values, as the maximum of unbiased estimates is itself biased. This can lead to suboptimal policies being learned.
- Double Q-Learning is a proposed solution to address the maximization bias, by maintaining two separate Q-functions and using one for decision making and the other for value estimation.

## Comparison of Sarsa and Q-Learning

**Table:** Comparing the key differences between Sarsa and Q-Learning:

| Feature | Sarsa | Q-Learning |
| --- | --- | --- |
| Update Rule | Uses the actual next action taken | Uses the maximum expected future reward |
| Update Timing | Waits for the next action to be taken | Can update earlier, without waiting for the next action |
| Policy Update | Updates the policy for multiple states | Only updates the policy for the current state |
| Maximization Bias | Less susceptible | More susceptible |

---

# Value Function Approximation

This section covers the concept of value function approximation, which is used to represent the value of states or state-action pairs when the state or action space is too large to store explicitly. The notes discuss the benefits and challenges of using function approximation, as well as different types of function approximators that can be used, with a focus on linear value function approximation.

## Key Points

- **Motivation for Value Function Approximation**
    - Many real-world problems have enormous state and action spaces, making it infeasible to store a table of values for each state or state-action pair.
    - Function approximation allows for generalization, enabling the agent to make good decisions even in states it has not encountered before.
- **Types of Function Approximation**
    - Function approximation can take many forms, such as deep neural networks, polynomials, or other parameterized functions.
    - The choice of function approximator involves a trade-off between representational capacity, memory/computation requirements, and the amount of data needed to learn.
- **Linear Value Function Approximation**
    - In linear value function approximation, the value function is represented as a linear combination of features, with weights as the parameters.
    - This provides a compact representation and can be optimized using gradient-based methods.
- **Monte Carlo Policy Evaluation with Linear Function Approximation**
    - The goal is to estimate the value function for a given policy using Monte Carlo sampling of returns.
    - The update rule compares the observed return from a state to the current value function estimate, weighted by the feature vector for that state.
    - This can be done incrementally, updating the weights after each episode, or in batch, solving a linear regression problem using all the data.
- **Convergence Guarantees**
    - For on-policy Monte Carlo policy evaluation with linear function approximation, the weights will converge to the best possible linear approximation of the true value function.
    - However, this does not guarantee that the approximation will be perfect, as the true value function may not be representable by the linear function approximator.

## Table

| Benefit | Challenge |
| --- | --- |
| Reduced memory requirements | Potential for poor approximation quality |
| Reduced computation | Bias-variance trade-off in function approximator choice |
| Reduced experience needed | Convergence issues when combining with control |

## Quotes

> "We're gonna need to be able to generalize from our prior experience. So that even if we end up in a state action pair that we've never seen exactly before, it's like a slightly different set of pixels than we've ever seen before that we're still going to be able to make good decisions and that's going to require generalization."
> 

> "These choices of representation are defining sort of hypothesis classes, they're defining spaces over which you can represent policies and value functions."
> 

> "If you choose a really restricted representational capacity, you're going to have a bias forever because you're just not going to be able to represent the true function."
> 

# Temporal Difference (TD) Learning with Function Approximation

This set of notes covers the key concepts and techniques related to using temporal difference (TD) learning with function approximation, including policy evaluation and control. It discusses the benefits and challenges of combining function approximation, bootstrapping, and off-policy learning, as well as the theoretical properties and practical considerations of these approaches.

## Key Terms and Important Points

- **Function Approximation**: Representing the value function or action-value function using a parameterized function, such as a linear combination of features, rather than a table.
- **Bootstrapping**: Using an estimate of the value function to update the value function, rather than waiting for the full return.
- **Sampling**: Approximating the expected value by sampling a single transition, rather than summing over all possible next states.
- **On-policy vs. Off-policy**: On-policy methods use data generated by the policy being evaluated/learned, while off-policy methods can use data from a different policy.
- **Tabular Representation**: Representing the value function or action-value function using a separate parameter for each state or state-action pair.
- **TD Zero**: The most common variant of temporal difference learning, which updates the value function based on the immediate reward and estimated value of the next state.
- **Deadly Triad**: The combination of function approximation, bootstrapping, and off-policy learning can lead to instability and divergence.

## Policy Evaluation with Function Approximation

- With function approximation, we represent the value function using a parameterized function, such as a linear combination of features.
- We can use TD learning to update the function parameters, using the TD target (reward + γ * estimated value of next state) as the target.
- The TD update rule looks similar to the Monte Carlo update, but it bootstraps the value estimate rather than using the full return.
- In the tabular case, both Monte Carlo and TD zero converge to the optimal value function, with TD zero converging to within a constant factor of the minimum mean squared error.
- With linear function approximation, TD zero converges to within 1/(1-γ) of the minimum mean squared error.

## Control with Function Approximation

- For control, we represent the action-value function Q(s, a) using function approximation.
- We can use methods like Monte Carlo, Sarsa, and Q-learning, but now we're interleaving policy evaluation and policy improvement.
- The combination of function approximation, bootstrapping, and off-policy learning can lead to instability and divergence, known as the "deadly triad".
- In the tabular case, everything converges nicely, but with linear function approximation, there can be oscillations, and with nonlinear function approximation, convergence is not guaranteed.
- Key factors that affect convergence and performance are the objective function and the function representation used.

## Comparison of Approaches

| Approach | Tabular Representation | Linear Function Approximation |
| --- | --- | --- |
| Monte Carlo Policy Evaluation | Converges to optimal value function | Converges to minimum mean squared error |
| TD Zero Policy Evaluation | Converges to optimal value function | Converges to within 1/(1-γ) of minimum mean squared error |
| Control (e.g., Q-learning) | Converges | Can oscillate or diverge |

## Practical Considerations and Recent Developments

- The choice of objective function and function representation can significantly impact convergence and performance.
- Recent work has produced algorithms with convergence guarantees, even for nonlinear function approximation (e.g., Batch-Constrained Deep Q-Learning).
- Understanding the issues that can arise with the "deadly triad" is crucial for applying reinforcement learning with function approximation effectively.

---

# Deep Learning and Deep Reinforcement Learning

- This class will cover deep learning and deep reinforcement learning
- There is a broad spectrum of background knowledge on deep learning among the students
- Some students have used TensorFlow or PyTorch before, others have not
- This week's sessions will provide more background on deep learning, which is necessary for Homework 2
- The default project for the class will be released by the end of the next day, and students can choose to do their own project or the default project

## Linear Value Function Approximation

- Last week, we discussed linear value function approximation
- The value function is represented as a dot product between features describing the state and weights
- The objective is to minimize the mean squared error between the estimated value and the true value
- We can use either the return from a full episode or a bootstrapped return as the target value
- The derivative of the loss function with respect to the weights is simply the features times the prediction error

## Limitations of Linear Value Function Approximation

- Linear value function approximation works well if you have the right set of features
- However, finding the right set of features can be challenging
- Kernel-based approaches like k-nearest neighbors can have stronger convergence properties but don't scale well with high-dimensional inputs

## Deep Neural Networks

- Deep neural networks are a flexible function approximation class that can represent any function
- They are composed of multiple layers of functions, where each layer applies a linear transformation followed by a non-linear activation function
- The parameters of the deep neural network can be trained using stochastic gradient descent and backpropagation
- Convolutional neural networks are a specialized type of deep neural network that exploit the structure of images and can be more efficient than fully connected networks for high-dimensional inputs

## Deep Q-Learning

- Deep Q-learning uses a deep neural network to represent the Q-function
- To address the instability issues of function approximation in reinforcement learning, deep Q-learning uses:
    - Experience replay: Storing past experiences in a replay buffer and sampling from it during training
    - Fixed Q-targets: Periodically freezing the target network used to compute the TD target

## Key Points

- Deep neural networks and convolutional neural networks are powerful function approximators that can be used in reinforcement learning
- Deep Q-learning addresses the instability issues of function approximation through experience replay and fixed Q-targets
- The Atari game results demonstrated the potential of deep reinforcement learning, even though theoretical guarantees are still limited

### Table: Comparison of Function Approximation Approaches

| Approach | Advantages | Disadvantages |
| --- | --- | --- |
| Linear Value Function Approximation | - Simple
- Can work well with the right features | - Finding the right features can be challenging
- Limited representational power |
| Kernel-based Approaches (e.g., k-nearest neighbors) | - Strong convergence properties | - Scales poorly with high-dimensional inputs
- Computationally and memory intensive |
| Deep Neural Networks | - Universal function approximator
- Can learn features automatically | - Lack of theoretical guarantees
- Potential instability issues |

# Reinforcement Learning with Deep Q-Networks (DQN)

## Description

This set of notes covers the key concepts and techniques behind Deep Q-Networks (DQN), a reinforcement learning algorithm that combines deep learning with Q-learning to achieve human-level performance on a variety of Atari video games. The notes explore the core ideas of DQN, including experience replay, fixed Q-targets, and the benefits of these approaches. Additionally, the notes discuss several extensions and improvements to the original DQN algorithm, such as Double DQN, Prioritized Experience Replay, and Dueling DQN.

## Key Points

- **Experience Replay**
    - Stores transitions (state, action, reward, next state) in a replay buffer
    - Samples random mini-batches from the replay buffer for training, instead of using a single transition
    - Allows the agent to reuse data, improving data efficiency and stability
- **Fixed Q-Targets**
    - Maintains a separate "target network" with fixed weights to compute the target Q-values
    - Helps stabilize the learning process by reducing the noise in the target values
    - Prevents the target values from changing too quickly during training
- **Benefits of Experience Replay and Fixed Q-Targets**
    - Significantly improves performance on Atari games compared to the original Q-learning algorithm
    - Allows the agent to learn more efficiently and discover effective strategies to maximize the reward
- **Double DQN**
    - Maintains two separate Q-networks: one for action selection and one for evaluation
    - Addresses the maximization bias in the original DQN algorithm
    - Improves performance on many Atari games
- **Prioritized Experience Replay**
    - Prioritizes the sampling of transitions with high temporal-difference (TD) error
    - Allows the agent to focus on the most informative experiences during training
    - Can lead to exponential improvements in convergence rates compared to uniform sampling
- **Dueling DQN**
    - Separates the estimation of state value and action advantage in the network architecture
    - Helps the agent focus on learning which actions are better or worse in a given state
    - Provides additional performance gains on top of Double DQN and Prioritized Experience Replay

## Table: Comparison of DQN Variants

| Variant | Description | Performance Improvement |
| --- | --- | --- |
| DQN | Original Deep Q-Network algorithm | - |
| Double DQN | Addresses maximization bias in DQN | Significant |
| Prioritized Experience Replay | Prioritizes sampling of high TD-error transitions | Substantial |
| Dueling DQN | Separates state value and action advantage estimation | Additional performance gain |

## Practical Tips

- Start with a linear Q-learning implementation before moving to the Atari domain
    - Helps with understanding and debugging the core Q-learning algorithm
    - Avoids the time-consuming process of training on Atari games directly
- Consider the impact of the frequency of updating the target network
    - There is a trade-off between propagating information faster and maintaining stability
    - Updating the target network less frequently can improve stability but slow down information propagation

---

# Deep Reinforcement Learning: Imitation Learning and Large State Spaces

- Recap of DQN algorithm:
    - Combines Q-learning with deep neural networks as function approximators
    - Key algorithmic changes: experience replay and fixed Q-targets
- Extensions to DQN:
    - Double DQN
    - Prioritized Experience Replay
    - Dueling Networks

## Double DQN

- Addresses the issue of maximization bias in DQN
- Maintains two sets of weights: one for action selection and one for action evaluation
- Allows for faster information propagation by frequently updating both networks

## Prioritized Experience Replay

- Prioritizes samples based on the size of the DQN error
- Can provide exponential speed-up in convergence, but computationally intensive

## Dueling Networks

- Separates the representation of state value and action advantage
- Allows for different features to be used for value and advantage estimation
- Not a unique decomposition, but a heuristic approach

## Practical Tips

- Actively encourage building up Q-learning representation before applying to Atari
- Try different forms of loss functions
- Learning rate is important, but the Adam optimizer can help

## Rainbow

- Combines multiple extensions (Double DQN, Prioritized Replay, Dueling) into a single algorithm
- Significant performance improvements over individual extensions

## Limitations of Model-Free Deep RL

- Hardness results indicate pathological MDPs where a lot of data is needed
- Motivation for exploring other approaches, such as imitation learning

## Imitation Learning

- Problem setup: unknown transition model, no reward function, but access to expert demonstrations
- Two main approaches:
    1. Behavior Cloning: Directly learn the expert's policy
    2. Inverse Reinforcement Learning: Recover the expert's reward function

## Behavior Cloning

- Treat as a supervised learning problem, learning a mapping from states to actions
- Issue: Compounding errors due to the distribution shift between training and test data
- DAGGER algorithm: Iteratively aggregate data by querying the expert for actions in encountered states

## Inverse Reinforcement Learning

- Goal: Recover the reward function that makes the expert's policy optimal
- Challenge: Reward function is not unique without additional assumptions
- Approach: Assume a linear value function and linear reward function

## Linear Value Function and Reward Approximation

- Express the value function as a linear combination of state features
- Define the "feature expectation" as the discounted state feature visitation frequencies
- Relate the expert's feature expectation to the optimal reward function

## Apprenticeship Learning

- Use the expert's feature expectation to learn a reward function that makes the expert's policy optimal

| Key Concepts | Description |
| --- | --- |
| Double DQN | Addresses maximization bias by using separate networks for action selection and evaluation |
| Prioritized Experience Replay | Prioritizes samples based on TD error to improve sample efficiency |
| Dueling Networks | Separates value and advantage estimation to capture different relevant features |
| Rainbow | Combines multiple DQN extensions into a single algorithm for significant performance gains |
| Behavior Cloning | Directly learns the expert's policy through supervised learning |
| DAGGER | Iteratively aggregates data by querying the expert for actions in encountered states |
| Inverse Reinforcement Learning | Recovers the reward function that makes the expert's policy optimal |
| Linear Value Function Approximation | Represents the value function as a linear combination of state features |
| Feature Expectation | Discounted state feature visitation frequencies under a policy |
| Apprenticeship Learning | Uses the expert's feature expectation to learn an optimal reward function |

# Imitation Learning and Inverse Reinforcement Learning

## Matching State Distributions

- The value of the optimal policy is directly related to the distribution of states you get under it times the weight vector (W)
- The value of the optimal policy has to be higher than the value of any other policy using the same weight vector (W)
- The discounted distribution of states you reach under a policy should be close to the distribution of states from the expert demonstrations
- If you can match the feature expectations or distributions of states, then you'll find a value that's very similar to the true value, no matter what the true reward function is

## Bounding the Reward Function

- The constraint on the affinity norm over W is there to ensure that errors are bounded and things don't explode during the backups and approximations
- Keeping the reward function bounded is important for convergence, even with discounting
- Unbounded rewards can make the problem much worse in terms of convergence

## Limitations of State Distribution Matching

- Matching state distributions doesn't necessarily identify the true reward function
- It just ensures that the policy matches the expert's, even if the reward function is unknown
- This approach gives up on learning the true reward function and instead focuses on matching the expert's policy

## Extensions and Alternatives

- Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) - Picks the most uncertain reward function that still respects the constraints of the expert data
- Generative Adversarial Inverse Imitation Learning (GAIL) - Uses a discriminator to compare the agent's generated trajectories to the expert's, without needing to explicitly model the state distribution
- Bayesian approaches can incorporate prior knowledge about the reward function
- Combining inverse RL with online RL can allow the agent to outperform the expert, but introduces challenges around safe exploration

## Key Takeaways

- Imitation learning and inverse RL are powerful tools, but have limitations in recovering the true reward function
- Matching state distributions can be sufficient to learn the expert's policy, even without knowing the reward
- Extensions like MaxEnt IRL and GAIL address some of the limitations by using more flexible approaches
- Incorporating prior knowledge through Bayesian methods or combining with online RL are active areas of research

| Approach | Key Idea | Advantages | Limitations |
| --- | --- | --- | --- |
| State Distribution Matching | Match the distribution of states visited by the expert | Simple, can guarantee matching the expert's policy | Does not necessarily recover the true reward function |
| MaxEnt IRL | Choose the most uncertain reward function that still respects the expert data | Principled way to handle reward function ambiguity | Still relies on modeling the state distribution |
| GAIL | Use a discriminator to compare the agent's trajectories to the expert's | Avoids explicit modeling of the state distribution | Can be more complex to implement and train |
| Bayesian Methods | Incorporate prior knowledge about the reward function | Can leverage domain expertise to guide the learning process | Requires specifying appropriate priors, which can be challenging |

---

# Policy Gradient Methods

Policy gradient methods are a popular approach in reinforcement learning for learning policies directly, rather than learning value functions. These methods parameterize the policy and perform gradient-based optimization to find the best policy parameters. This can be more effective in high-dimensional or continuous action spaces, and allows for learning stochastic policies.

## Key Points

- Policy gradient methods directly parameterize the policy function, rather than the value function
- Parameterizing the policy defines the space of policies that can be learned
- Policy gradient methods are model-free, not requiring knowledge of the environment dynamics or reward function
- Policy gradient methods only converge to a local optimum, not a global optimum
- Policy gradient methods can be sample-inefficient, requiring a lot of data to estimate the gradient

## Advantages of Policy Gradient Methods

- Can be more effective in high-dimensional or continuous action spaces
- Allow learning of stochastic policies, which can be beneficial in certain environments
- Sometimes have better convergence properties than model-based approaches

## Disadvantages of Policy Gradient Methods

- Only converge to a local optimum, not a global optimum
- Can be sample-inefficient, requiring a lot of data to estimate gradients

## Policy Gradient Derivation

1. Define the policy value as the expected discounted sum of rewards under the policy
2. Rewrite the policy value in terms of the probability of trajectories and their rewards
3. Take the gradient of the policy value with respect to the policy parameters
4. Use the likelihood ratio trick to express the gradient without needing to know the environment dynamics
5. The final policy gradient estimator is the expected value of the reward times the gradient of the log probability of the trajectory

## Temporal Structure and Baselines

- The basic policy gradient estimator is unbiased but very noisy
- Techniques like temporal structure and baselines can be used to reduce the variance of the estimator

## Table: Comparison of Policy Gradient and Value-Based Methods

| Characteristic | Policy Gradient Methods | Value-Based Methods |
| --- | --- | --- |
| Parameterization | Policy function | Value function |
| Convergence | Local optimum | Global optimum (with some exceptions) |
| Sample Efficiency | Can be sample-inefficient | Can be more sample-efficient |
| Action Spaces | Effective in high-dimensional/continuous spaces | Better in discrete, low-dimensional spaces |
| Stochastic Policies | Can learn stochastic policies | Typically learn deterministic policies |

---

# Policy Gradient Methods

## Reorganizing the Sum

- We've reorganized the sum in a slightly different way, but it's done in a useful way.
- The second term on the bottom line represents the reward we get starting at time step t all the way to the end, which is just the return.
- This is familiar from our discussion of Monte Carlo methods, where we could look at the sum of rewards starting from a given state-action pair until the end of the episode.

## Derivative of the Policy Parameters

- We can rewrite the derivative with respect to theta as approximately 1/M sum over all trajectories of the derivative with respect to theta of the actual policy parameter times the return.
- This is a slightly lower variance estimate than before, as we're only needing to take the sum of the logs for some of the reward terms, rather than multiplying by the full sum of all the derivatives of the logs.
- This is saying that for every single reward, you only have to multiply it by the ones that are relevant for that particular reward, rather than the full trajectory in terms of the derivative of the policy parameters.

## The REINFORCE Algorithm

- REINFORCE is one of the most common reinforcement learning policy gradient algorithms.
- The algorithm is as follows:
    1. Initialize the policy parameters randomly.
    2. For each episode:
        - Sample a trajectory from the current policy.
        - For each time step in the trajectory:
            - Update the policy parameters using the gradient: `alpha * (derivative of log pi_theta(s,a) w.r.t. theta) * G_t`
            - Where `G_t`
                
                is the return from time step t onwards.
                
    3. Return the final policy parameters.
- This is an unbiased estimate of the gradient, but it is stochastic.
- There is no notion of state and actions in the same way as before, as we are directly updating the policy parameters.

## Parameterizing the Policy

- Common policy parameterizations include:
    1. **Softmax**: Exponential weighting of features, good for discrete action spaces.
        - The derivative of the log of the softmax policy is the features minus the expected features.
    2. **Gaussian**: Good for continuous action spaces.
        - The policy is parameterized by a mean (linear in state features) and a fixed variance.
        - The score function is the derivative of the Gaussian, which is (action - mean) / variance.
    3. **Deep Neural Networks**: A very common and flexible parameterization.

## Table of Policy Parameterizations

| Parameterization | Advantages | Derivative of Log Policy |
| --- | --- | --- |
| Softmax | Discrete action spaces | Features - Expected Features |
| Gaussian | Continuous action spaces | (Action - Mean) / Variance |
| Deep Neural Networks | Flexible and powerful | Computed via backpropagation |

---

# Policy Search in Reinforcement Learning

This set of notes covers the key concepts and techniques related to policy search methods in reinforcement learning. Policy search is a powerful approach for optimizing policies, especially in large state spaces, and can provide guarantees of convergence to a local optimum. The notes discuss the vanilla policy gradient algorithm, the use of baselines and value functions (critics) to reduce the variance of gradient estimates, and strategies for ensuring monotonic improvement of the policy during the optimization process.

## Key Topics

- Vanilla Policy Gradient Algorithm
- Baselines and Advantage Functions
    - Using value functions (critics) to estimate advantages
    - Bias-variance tradeoffs of different return estimation methods
- Ensuring Monotonic Improvement
    - Expressing value in terms of advantage over old policy
    - Choosing step sizes to guarantee monotonic improvement

## Notes

### Vanilla Policy Gradient Algorithm

- Policy is parameterized by θ, and we want to find the optimal θ that maximizes the policy's value
- Vanilla policy gradient algorithm:
    1. Collect a set of trajectories using the current policy
    2. For each time step t in each trajectory i:
        - Compute the return G_i,t (sum of rewards from t to end of episode)
        - Compute the advantage estimate: A_i,t = G_i,t - b(s_t)
            - b(s_t) is a baseline function that depends only on the state
    3. Update the policy parameters θ using the gradient:
    
    $$
    Δθ ∝ (1/M) Σ_i Σ_t ∇_θ log π_θ(a_i,t|s_i,t) A_i,t
    $$
    

### Baselines and Advantage Functions

- Baselines help reduce the variance of the gradient estimate without introducing bias
- Baselines can be implemented as an estimate of the value function V_π(s)
- Different return estimation methods can be used:
    - Monte Carlo returns (unbiased, high variance)
    - TD-style returns (biased, lower variance)
    - n-step returns (interpolate between MC and TD)
- Using a critic (parameterized value function) to estimate the advantage:
    - Advantage A(s,a) = Q(s,a) - V(s)
    - Gradient: Δθ ∝ (1/M) Σ_i Σ_t ∇_θ log π_θ(a_i,t|s_i,t) A_i,t

### Ensuring Monotonic Improvement

- Goal: Ensure V_π_{i+1} ≥ V_π_i (monotonic improvement)
- Rewrite value function in terms of advantage over old policy:V_π_tilde = V_π + Σ_t γ^t A_π(s_t, a_t)
- This allows us to choose step sizes that guarantee monotonic improvement
- Key challenge: Data used to estimate the gradient is from the old policy, so it's an off-policy problem
- Strategies for ensuring monotonic improvement:
    - Careful step size selection (e.g., line search)
    - Explicitly bounding the change in policy (e.g., trust region methods)

### Table: Comparison of Return Estimation Methods

| Method | Bias | Variance |
| --- | --- | --- |
| Monte Carlo Returns | Unbiased | High |
| TD-style Returns | Biased | Lower |
| n-step Returns | Interpolate between MC and TD | Adjustable |

---

---

# Policy Gradient Algorithms

- The class is nearing the end of the policy search section
- The midterm exam will be held on Wednesday, with a holiday on Monday
- The last homework assignment on policy search will be released this week
- After the midterm, the class will cover fast exploration and fast reinforcement learning

## Policy Gradient Algorithms

- Policy-based reinforcement learning aims to find a parameterized policy that can make good decisions in the environment
- The policy can be represented using techniques like softmax or deep neural networks
- The goal is to take gradients of the policy in order to learn a high-value policy
- The vanilla policy gradient algorithm:
    - Initialize the policy
    - Maintain a baseline
    - Run the current policy and collect trajectories of state-action-reward sequences
    - Estimate the gradient of the policy using the collected data

## Estimating the Value of the Current Policy

- When estimating the value of the current policy, we can use techniques like:
    - Monte Carlo estimates: Unbiased but high variance
    - Bootstrapping and function approximation: Can introduce bias to reduce variance
- Critic methods maintain explicit representations of both the policy and the value function
- The key challenge is determining how far to move along the gradient to update the policy

## Ensuring Monotonic Improvement

- The goal is to ensure monotonic improvement, where the new policy is expected to be better than the old policy
- This is challenging because we don't have data from the new policies we're considering
- We can express the value of a new policy in terms of the value of the current policy plus an advantage term
- This allows us to define a new objective function that can be optimized using only data from the current policy

## Theoretical Guarantees

- The new objective function provides a lower bound on the value of the new policy
- This lower bound can be used to guarantee monotonic improvement if the new policy's lower bound is higher than the current policy's value
- The lower bound is related to the total variation distance between the current and new policies

## Practical Considerations

- The total variation distance is difficult to compute, so we can use the KL divergence as an upper bound
- This leads to the definition of a new objective function that can be optimized while ensuring monotonic improvement

## Trust Region Policy Optimization (TRPO)

- TRPO is a popular policy gradient algorithm that builds on the ideas of conservative policy improvement
- TRPO defines a constraint on how much the new policy can differ from the old policy, using the KL divergence
- This allows TRPO to take larger steps along the gradient while still ensuring monotonic improvement

## Review of Key Concepts

### Reinforcement Learning Fundamentals

- Key features of reinforcement learning:
    - Agent collects its own data, which influences the policies it can learn
    - Censored data issue: Agent can't know about other lives that didn't live
- Formulating a problem as a reinforcement learning problem:
    - Define the state space, action space, dynamics, and reward model
    - Suggest an appropriate algorithm from the class
- Evaluating RL algorithms:
    - Criteria like bias, variance, computational complexity, sample efficiency

### Planning, Policy Evaluation, and Policy Iteration

- Planning: Knowing the model of the world, compute the optimal policy
- Policy evaluation: Estimate the value of a given policy
    - Dynamic programming, Monte Carlo, and temporal difference learning
- Policy iteration: Alternate between evaluating and improving the policy

### Model-Free Policy Evaluation

- Monte Carlo: Unbiased but high variance
- Temporal difference (TD) learning: Samples a single next state and bootstraps
- Handling non-Markovian domains:
    - Dynamic programming and Monte Carlo assume Markovian domains
    - TD learning makes Markovian assumptions

### Example: Random Walk Domain

- Compute the true value function for a simple random walk domain
- Compare the estimates from different algorithms (Monte Carlo, TD)

**Table: Comparison of Policy Evaluation Algorithms**

| Algorithm | Usable without Model | Handles Continuing Domains | Handles Non-Markovian Domains | Converges to True Value | Unbiased Estimates |
| --- | --- | --- | --- | --- | --- |
| Dynamic Programming | No | Yes | No | Yes | - |
| Monte Carlo | Yes | No | No | Yes | Yes |
| Temporal Difference | Yes | Yes | No | Yes | No |

---

# Reversing the Order of Data Updates in Machine Learning

- The order of updates can matter in terms of the final values computed, depending on the algorithm and initialization.
- It's important to convince yourself whether the order matters or not for a given problem.
- Compute a few updates to see the effect of reversing the order of the data.
- Spend time thinking about whether the order of updates matters in the given scenario.
- Compute one or two updates to see the impact of the order.
- The fact that order sometimes matters might lead you to believe it always does, but that's not the case.
- If the initial values are all zero, the order of updates can affect the final values computed.
- When updating B(C), the new value of V(C) can be used, since it has already been updated.
- Initializing values to be optimistic (non-zero) can help in some cases, but can also be challenging in deep neural networks.
- TD learning tends to be computationally efficient, but data efficiency can depend on factors like experience replay.
- It's important to be precise in stating assumptions, such as whether you're using the vanilla version of an algorithm or variations like experience replay.
- Q-learning is a model-free, bootstrapping technique that assumes a Markovian world.
- Q-learning can converge to the optimal Q-function under mild reachability assumptions, even when using data gathered by a different policy.
- An ε-greedy policy, which selects the best action with probability 1-ε and a random action with probability ε, is guaranteed to converge to the optimal policy in the tabular setting with infinite data.
- Monte Carlo estimation can be used in MDPs with large state spaces, but the number of data points per state may be low, requiring function approximation.
- Model-based reinforcement learning is not necessarily always more data-efficient than model-free approaches, as shown in recent research.
- In on-policy value function approximation, Monte Carlo methods converge to the best mean-squared error possible given the function approximation space, while TD learning converges to a constant factor of that.
- With linear value function approximation, there may be a fundamental gap between the true value function and the representable value function.
- Off-policy Q-learning with function approximation can diverge, even with infinite data, due to the instability of the updates.
- Techniques like experience replay, fixed targets, double Q-learning, and prioritized replay have been developed to improve the stability and convergence of deep Q-learning.
- Initializing the function approximation parameters can impact whether Q-learning converges or diverges, but there is no formal characterization of this.
- In the on-policy case with a representation that can exactly represent the true value function, TD learning is guaranteed to find the true value function with sufficient data.
- This guarantee holds for both linear and nonlinear function approximation, as long as the representation is expressive enough.
- Imitation learning aims to learn policies by imitating expert demonstrations, either through behavior cloning or inverse reinforcement learning.
- Challenges in imitation learning include the distribution shift between the learner's policy and the expert's policy.
- Policy search methods, such as policy gradients, can be used to directly optimize parameterized stochastic policies.

## Table

| Technique | Convergence Guarantee |
| --- | --- |
| Monte Carlo (on-policy) | Converges to the best mean-squared error possible given the function approximation space |
| TD Learning (on-policy) | Converges to a constant factor of the best mean-squared error possible given the function approximation space |
| Q-Learning (off-policy with function approximation) | Can diverge, even with infinite data, due to instability of updates |

---

# Fast Reinforcement Learning

## Introduction

- We will be discussing fast reinforcement learning
- This comes after completing policy search and policy gradient
- The rest of the course will focus on projects

## Motivation for Fast Reinforcement Learning

- Many real-world applications involve reinforcement learning with people, such as education, healthcare, and marketing
- Data from interacting with people is expensive and limited, so we need sample-efficient algorithms
- Most current reinforcement learning techniques focus more on computational efficiency rather than sample efficiency

## Computational Efficiency vs. Sample Efficiency

- Computational efficiency is important for real-time applications like robotics
- Sample efficiency is crucial when interacting with people, as we don't want to randomly experiment on them
- Existing techniques like Q-learning are not sample-efficient enough for many applications

## Measuring Algorithm Quality

- Convergence: Does the algorithm converge at all?
- Asymptotic optimality: Does it converge to the optimal policy?
- Sample efficiency: How quickly does it get to the optimal policy?
- Regret: The difference between the optimal policy's performance and the algorithm's performance

## Outline for the Lecture Series

- Today and next lecture:
    - Introduce multi-armed bandits
    - Define regret
    - Discuss optimism under uncertainty
    - Explore Bayes and Thompson sampling
- Future lectures:
    - Extend to Markov Decision Processes (MDPs)
    - Discuss evaluation frameworks and approaches for fast reinforcement learning

## Multi-Armed Bandits

**Description:**

- Multi-armed bandits are a subset of reinforcement learning problems
- Each "arm" (action) has an unknown reward distribution
- The goal is to maximize the cumulative reward by selecting arms wisely

**Key Points:**

- Arms have unknown reward distributions, often assumed to be bounded between 0 and 1
- The agent's goal is to maximize the cumulative reward by selecting the best arm
- The value of an arm is its expected reward, denoted as Q(a)
- The optimal arm is the one with the highest expected reward, Q(a*)
- Regret is the difference between the reward of the optimal arm and the reward of the arm selected

## Regret

**Definition:**

- Regret is the opportunity loss for one step, i.e., the difference between the reward of the optimal arm and the reward of the arm selected.
- Total regret is the sum of the regret over all time steps.

**Key Points:**

- The agent does not know the true expected reward of each arm (Q(a)) in advance.
- The agent can only observe samples of rewards from the unknown distributions.
- Regret is a way to measure the performance of the agent in the absence of full information.

## Optimism Under Uncertainty

**Description:**

- Optimism under uncertainty is an approach where the agent selects actions that might have high value.
- The agent maintains upper confidence bounds (UCBs) for the expected reward of each arm.
- The agent selects the arm with the highest UCB.

**Key Points:**

- UCBs are constructed using Hoeffding's inequality to ensure they hold with high probability.
- The UCB for an arm depends on the empirical mean reward and the number of times the arm has been pulled.
- Selecting the arm with the highest UCB balances exploration (learning about arms) and exploitation (selecting the best arm).

**Table: UCB Algorithm**

| Step | Action |
| --- | --- |
| 1. Initialization | Pull each arm once to get initial rewards. |
| 2. For t = 1 to T: |  |
| a. Compute UCBs for all arms |  |
| b. Select arm A_t = argmax_a U_t(a) |  |
| c. Observe reward r_t |  |
| d. Update UCBs for all arms |  |

## Regret Analysis of UCB

**Key Points:**

- We want to bound the probability that the UCBs fail to hold at any step.
- If the UCBs always hold, then the selected arm's UCB is greater than the true optimal arm's value.
- This allows us to bound the regret of the UCB algorithm.
- The regret grows sublinearly with the number of time steps, which is better than the linear regret of naive exploration strategies.

---

# Optimism Under Uncertainty: Regret Bounds for Upper Confidence Bound Algorithms

This set of notes covers the concept of optimism under uncertainty and how it can be used to derive regret bounds for upper confidence bound (UCB) algorithms in the context of multi-armed bandit problems. The key ideas discussed include:

- Defining the upper confidence bound for each arm and how it relates to the true mean value of the optimal arm
- Proving that the upper confidence bound of the selected arm is higher than the true mean of the optimal arm, if the confidence bounds hold
- Deriving a high-probability regret bound for the UCB algorithm by bounding the difference between the upper confidence bound and the true mean

## Key Points

- **Upper Confidence Bound (UCB)**
    - The upper confidence bound of an arm is defined as the empirical mean of that arm plus a term that depends on the number of times the arm has been pulled and a confidence parameter
    - If the confidence bounds hold, the upper confidence bound of the selected arm is higher than the true mean of the optimal arm
- **Regret Bound Derivation**
    - Regret is defined as the sum of differences between the true mean of the optimal arm and the true mean of the selected arm
    - By adding and subtracting the upper confidence bound of the selected arm, the regret can be bounded in terms of quantities that are known or can be bounded
    - The probability that the confidence bounds hold on all time steps is at least 1 - 2Mδ, where M is the number of arms and δ is the confidence parameter
    - The regret is bounded by a term that grows sublinearly with the number of time steps, specifically as √(T log(T²/δ))

## Table

| Notation | Description |
| --- | --- |
| `A_t` | The arm selected at time step t |
| `a*` | The optimal arm |
| `Q(a)` | The true mean reward of arm a |
| `Q̂(a)` | The empirical mean reward of arm a |
| `u_t(a)` | The upper confidence bound of arm a at time step t |
| `δ` | The confidence parameter |
| `M` | The number of arms |
| `T` | The number of time steps |
| `Regret(T)` | The regret after T time steps |

## Quotes

> "If the confidence bounds hold, then whatever arm you selected its upper confidence bound is higher than the value of the true arm. The real value of the true arm."
> 

> "This says if we could get it and we'll see shortly why that matters. But kind of intuitively this says if the confidence bounds hold, then we know that separate confidence bound of the arm we select is going to be better than the optimal arm."
> 

> "We're now gonna write that down in terms of what's the probability that the upper confidence bounds do hold on all time steps."
> 

> "We're gonna end up having a high probability regret bound that says with probability at least one minus two M delta, you're gonna get a small regret."
> 

---

# Optimism Under Uncertainty and Bayesian Bandits

## Optimism Under Uncertainty

- Recap of Bandits and Regret
    - Bandits are a simplified version of Markov Decision Processes (MDPs)
    - Rewards are stochastic and unknown, and the goal is to maximize cumulative rewards over time
    - Regret is the difference between the expected reward of the optimal action and the expected reward of the action taken
- Optimism Under Uncertainty
    - Estimate an upper confidence bound on the potential expected reward of each arm
    - Select the arm with the highest upper confidence bound
    - This can lead to two outcomes:
        1. The selected arm is the optimal arm, resulting in zero regret
        2. The selected arm is not optimal, causing the upper confidence bound to decrease as more information is gained
    - This idea was first introduced by Leslie Kale in the 1990s
    - The upper confidence bound algorithm can achieve logarithmic regret, which is better than the linear regret of greedy algorithms
    - The regret bound depends on the gaps between the expected rewards of the optimal arm and the suboptimal arms
- Toy Example: Broken Toe Treatments
    - Comparing the behavior of Upper Confidence Bound (UCB) and ε-greedy algorithms
    - UCB selects only the arms with the highest upper confidence bounds, while ε-greedy explores all arms with a small probability
    - Calculating the regret for each action taken by the UCB algorithm

## Bayesian Bandits

- Bayesian Approach to Bandits
    - Assume a parametric distribution for the rewards of each arm
    - Use Bayesian inference to update the posterior distribution over the parameters as more data is observed
    - This allows for more informed exploration and exploitation decisions
- Thompson Sampling
    - Sample a parameter value from the posterior distribution for each arm
    - Select the arm with the highest expected reward under the sampled parameters
    - This implicitly implements probability matching, selecting arms proportional to the probability they are optimal
    - Conjugate priors, such as the Beta distribution for Bernoulli rewards, allow for efficient Bayesian updates
- Toy Example: Broken Toe Treatments (Continued)
    - Initialize a Beta(1,1) prior over the Bernoulli parameters for each arm
    - Sample parameters from the priors and select the arm with the highest expected reward
    - Update the posterior distributions as rewards are observed
    - Observe how the posterior for the optimal arm becomes more concentrated around the true parameter value over time

---

# Optimistic Initialization and Exploration Bonuses in Markov Decision Processes

This set of notes covers the use of optimistic initialization and exploration bonuses in Markov Decision Processes (MDPs), as discussed in the video transcript. It explores different approaches to achieving probably approximately correct (PAC) performance in MDPs, including optimistic initialization, model-based methods, and the use of reward bonuses.

## Key Points

### Optimistic Initialization

- Initializing all Q-values to `R_max / (1 - γ)`
    
    is guaranteed to be optimistic
    
- This can encourage systematic exploration, but there are no guarantees on performance
- Ear and Mansour (2002) showed that Q-learning with this initialization can be PAC, but the initialization is extremely large (exponential in the number of time steps)
- More recent work by Chi Jin et al. has shown that model-free Q-learning with less optimistic initialization and careful learning rates can also achieve good regret bounds

### Model-Based Approaches

- One approach is to be extremely optimistic until confident in the empirical estimates of the dynamics and reward model
- Another approach is to be optimistic given the current information, using confidence sets on the dynamics and reward models
- Reward bonuses that depend on the agent's experience can be a practical alternative to explicit confidence sets, especially in the function approximation setting

### Model-Based Interval Estimation with Exploration Bonuses

- Initialize counts and cumulative rewards to 0, and Q-values to `1 / (1 - γ)`
- Define an empirical transition model and reward model based on the observed counts and rewards
- Compute Q-values using value iteration, with an exploration bonus term that decreases as more data is collected
- This algorithm is PAC, with the number of non-optimal actions bounded by a polynomial function of the problem parameters

| Algorithm | Optimistic Initialization | Exploration Bonuses | PAC Guarantee |
| --- | --- | --- | --- |
| Q-learning (Ear and Mansour 2002) | `R_max / (1 - γ)` | No | Yes, but initialization is extremely large |
| Model-free Q-learning | Less optimistic | No | Yes, with careful learning rates |
| Model-Based Interval Estimation | No | Yes | Yes, with polynomial bound on non-optimal actions |

---

# Exploration vs. Exploitation in Reinforcement Learning

## Reaching Unknown State-Action Pairs

- If the probability of reaching an unknown state-action pair is small, the agent is near-optimal on the known state-action pairs.
- If the probability of reaching an unknown state-action pair is large, the agent can only visit it a bounded number of times due to the pigeonhole principle.
- This means the agent will eventually know all state-action pairs or be acting near-optimally.
- The agent is either not reaching unknown parts of the state-action space (and making good decisions) or is reaching them and gaining new information.

## PAC-MDP and Regret Analysis

- Recent work has focused on improving PAC-MDP and regret analysis approaches.
- The goal is to have analysis that is problem-dependent, so the algorithm requires less data to learn good decisions if it has more structure.

## Bayesian Reinforcement Learning

### Bayesian Bandits

- Assume a parametric model of how rewards are distributed.
- Maintain a posterior distribution over the rewards and use it to guide exploration.
- Thompson sampling selects actions with probability proportional to their optimality.
- Conjugate priors (e.g., beta distribution for Bernoulli rewards) allow efficient posterior updates.

### Bayesian MDPs

- Maintain a posterior distribution over the MDP model (both transitions and rewards).
- Sample an MDP from the posterior, solve it, and act optimally with respect to the sampled MDP.
- This is known as Thompson sampling for MDPs or posterior sampling for MDPs.
- Researchers like Ben Van Roy, Dan Russo, and Ian Osband have done notable work in this area.

## Generalization and Exploration

### Challenges in Large State Spaces

- In large state spaces (e.g., pixel-based), the counts-based approach from tabular settings does not scale.
- Need a way to quantify uncertainty that generalizes across similar states.
- Model-based approaches have struggled due to compounding model errors, so model-free approaches are more promising.

### Exploration Bonuses in Model-Free RL

- Idea: Add a bonus term to the Q-learning update to encourage exploration of uncertain state-action pairs.
- Approaches to compute the bonus:
    - Density models to estimate state/state-action visitation counts
    - Hash-based methods to track counts over a compressed state space
- Bonuses can be computed at the time of visit or updated retrospectively in the replay buffer.

### Empirical Results

- Exploration bonuses have shown significant improvements on hard exploration tasks like Montezuma's Revenge.

### Thompson Sampling in Model-Free RL

- Sampling over representations (e.g., dynamic state aggregation) and parameters.
- Challenges in defining a posterior over possible Q-functions in the model-free setting.
- Approaches like Bootstrap DQN and Bayesian linear regression on top of a fixed embedding have shown promise.

| Criteria | Description |
| --- | --- |
| PAC-MDP | Probably Approximately Correct MDP - Bounds the number of steps where the agent is not near-optimal. |
| Regret | Measures the cumulative difference between the agent's rewards and the optimal rewards. |
| Exploration Bonus | Additional reward term added to the Q-learning update to encourage exploration of uncertain state-action pairs. |
| Thompson Sampling | Sampling from a posterior distribution over models/Q-functions and acting optimally with respect to the sampled model. |

---

# Safe Batch Reinforcement Learning

## Description

This set of notes covers the topic of safe batch reinforcement learning, which involves using historical data to estimate the value of alternative policies and make confident decisions about deploying new policies in high-stakes scenarios. The key concepts discussed include off-policy policy evaluation, high-confidence policy evaluation, and safe policy improvement.

## Key Points

### Off-Policy Policy Evaluation

- Importance sampling is a technique to estimate the value of an alternative policy (evaluation policy) using data collected under a different policy (behavior policy)
- The importance sampling estimator is unbiased and consistent, but requires the behavior policy to have non-zero probability for any action the evaluation policy might take (coverage/support assumption)
- Per-decision importance sampling can be used to reduce variance by only multiplying importance ratios up to the time the reward was received
- Weighted importance sampling renormalizes the importance weights to be between 0 and 1, introducing bias but reducing variance

**Table: Comparison of Importance Sampling Techniques**

| Technique | Bias | Variance |
| --- | --- | --- |
| Importance Sampling | Unbiased | High |
| Weighted Importance Sampling | Biased | Lower |

### High-Confidence Policy Evaluation

- Using control variates (e.g., a value function baseline) can further reduce the variance of the policy evaluation estimate
- The goal is to obtain tight upper and lower bounds on the value of a policy before deploying it in high-stakes scenarios

### Safe Policy Improvement

- The safe policy improvement problem is to find a new policy that is guaranteed to be better than the current policy with high confidence
- This requires being able to estimate the value of alternative policies accurately and quantify the uncertainty in those estimates
- Subtracting a control variate (e.g., a value function baseline) from the importance sampling estimator can reduce the variance without changing the expected value

---

# Monte Carlo Tree Search

## Description

Monte Carlo Tree Search (MCTS) is a simulation-based search algorithm used in reinforcement learning and game AI. It combines the strengths of model-based and model-free reinforcement learning approaches to make efficient decisions in complex environments. MCTS is particularly useful for games like Go, where the state space is too large for traditional planning methods.

## Key Points

- **Model-Based Reinforcement Learning**: MCTS leverages a model of the environment to simulate possible future trajectories and evaluate the expected value of actions.
- **Partial Tree Expansion**: Instead of fully expanding an exponentially growing search tree, MCTS selectively expands the most promising parts of the tree based on an upper confidence bound (UCB) strategy.
- **Rollout Policy**: When reaching unexplored parts of the tree, MCTS uses a rollout policy (often a simple, random policy) to simulate the remainder of the trajectory.
- **Backup and Action Selection**: MCTS backs up the rewards from the simulated trajectories to update the value estimates at each node in the tree. The action with the highest estimated value is then selected.
- **Iterative Refinement**: The MCTS process is repeated multiple times, with the tree gradually expanding and the value estimates becoming more accurate.

## Table: Comparison of MCTS and Expectimax Tree Search

| Feature | MCTS | Expectimax Tree Search |
| --- | --- | --- |
| Computational Complexity | Lower | Higher |
| Partial Tree Expansion | Yes | No |
| Rollout Policy | Used for unexplored parts | Not applicable |
| Backup and Action Selection | Averaging rewards | Expectation and maximization |
| Iterative Refinement | Yes | No |

# Reinforcement Learning in Game AI

## Understanding Reinforcement Learning (RL)

- RL is fundamentally different from supervised learning and planning
    - RL agents gather their own data and make decisions in the world, unlike the i.i.d. assumption in supervised learning
    - RL relies on the data gathered about the world to make decisions, unlike planning which has full knowledge of the environment
- Formulating an RL problem is crucial
    - Deciding how to represent the state space can have a huge impact on the difficulty of solving the problem
    - There are tradeoffs between function approximation and sample efficiency that must be considered

## Monte Carlo Tree Search (MCTS) for Go

- MCTS is a powerful technique for planning in large state spaces like the game of Go
    - The rules and reward structure of Go are known, but the search space is incredibly large
    - MCTS builds a search tree by selectively expanding promising parts of the tree through random rollouts
- Key aspects of MCTS for Go
    - Start with a single state and randomly sample actions, following a default policy to the end of the game
    - Keep track of the outcomes (wins/losses) for each state-action pair
    - Use the collected statistics to guide the tree expansion, favoring actions that have performed well
    - Repeat this process until the computational budget is exhausted
    - Then use the tree to select the best action at the root
- The importance of self-play
    - Using the current agent as the opponent allows the agent to bootstrap and learn from its own progress
    - Self-play helps address the sparse reward problem and provides a natural curriculum for learning

## Key Takeaways from the Course

1. Understanding the unique features of RL compared to supervised learning and planning
2. Importance of problem formulation and representation in RL
3. Familiarity with common RL algorithms and their properties
4. Addressing the exploration-exploitation tradeoff in RL
5. Potential for RL in a wide range of applications beyond games
6. Importance of developing safe, fair, and accountable RL systems

## Comparison of RL to Other Approaches

| Approach | Key Differences |
| --- | --- |
| Supervised Learning | RL agents gather their own data, unlike the i.i.d. assumption in supervised learning |
| Planning | RL relies on the data gathered about the world to make decisions, unlike planning which has full knowledge of the environment |

# Variational Autoencoders: Compressing High-Dimensional Data

## Introduction to Autoencoders

- Autoencoders are neural networks that compress high-dimensional input data into a lower-dimensional representation, and then try to reconstruct the original input from this compressed representation.
- The autoencoder consists of two main components:
    - **Encoder**: Compresses the input data into a smaller, bottleneck representation.
    - **Decoder**: Tries to reconstruct the original input from the bottleneck representation.
- The training process involves minimizing the reconstruction loss between the input and the output of the autoencoder.
- Autoencoders can be used for various tasks, such as image compression, denoising, and inpainting.

## Variational Autoencoders

- Variational Autoencoders (VAEs) are a type of autoencoder that map the input to a probability distribution, rather than a fixed vector.
- In a VAE, the bottleneck representation consists of two vectors: one representing the mean (μ) and one representing the standard deviation (σ) of the distribution.
- To train a VAE, the loss function has two components:
    1. **Reconstruction Loss**: Measures how well the decoder can reconstruct the original input from the sampled latent representation.
    2. **KL Divergence**: Measures how close the learned distribution is to a standard normal distribution (N(0, 1)).
- The reparameterization trick is used to enable backpropagation through the sampling operation in the VAE.

## Disentangled Variational Autoencoders

- Disentangled Variational Autoencoders (DVAEs) are a type of VAE that aim to learn a latent representation where each dimension corresponds to a distinct, interpretable factor of variation in the data.
- To achieve disentanglement, an additional hyperparameter (β) is added to the loss function that controls the strength of the KL divergence term.
- By increasing β, the DVAE is forced to use only a few latent dimensions to encode the input, resulting in a more disentangled representation.
- Disentangled representations can be useful for tasks like reinforcement learning, where the agent can learn useful behaviors on the compressed latent space.

## Applications and Tradeoffs

- VAEs and DVAEs have been applied to various domains, such as reinforcement learning, where the compressed latent representation can be used as input to the agent.
- There is a tradeoff when training VAEs and DVAEs:
    - If the latent space is not disentangled enough, the network may overfit to the training data and not generalize well.
    - If the latent space is too disentangled, the network may lose important high-dimensional details, which can hurt performance in certain applications.

| Comparison | Disentangled VAE | Normal VAE |
| --- | --- | --- |
| Latent Space Interpretation | Each latent dimension corresponds to a distinct, interpretable factor of variation | Latent dimensions are not easily interpretable |
| Generalization to Unseen Data | Better generalization due to the disentangled representation | May not generalize as well as the disentangled VAE |
| Reconstruction Quality | May lose some high-dimensional details due to the strong disentanglement constraint | Can preserve more high-dimensional details |