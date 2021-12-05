# Study for Reinforcement Learning with PyTorch


## Value-based

* **Q-Learning** 

  [Python test_code](./study/study_qlearning.py)

  ```
  tested_env : FrozenLake

  Update equation
    Q(S, A) <- Q(S, A) + alpha * (R + lambda * max_a( Q(next_S, :) ) - Q(S, A))
  
  ```

* **Deep Q-Network (DQN)**

  [Python test_code](./study/study_dqn_simple.py)

  ```
  tested_env : CartPole

  Main step for algorithms

    1. Randomly sampled a mini-batch from replay buffer.

    2. calculate the off-policy TD target with target network.
      R + gamma * max_a'(Q_Target(s',a'))

    3. Fit action-value function Q using MSE and RMSProp
  ```

* **Improvements**

  [E-Greedy Exploration methods](./study/nb_study_exploration_strategies.ipynb)

  ```
  1. Constant Epsilon-Greedy
  2. Linear Decay Epsilon-Greedy
  3. Exponential Decay Epsilon-Greedy
  ```

  [Loss functions](./study/nb_study_loss_functions.ipynb)
  ```
  Issues

  Mean Square Error (MSE or L2) penalizes large errors than small errors. Mean Absolute Error (MAE or L1) is a linear function but not differentiable at zero.
  The Huber loss has the advantages of both functions, penalizing the errors near zero and becoming linear for large errors.
  ```

* **Improved DQN**


  [Double DQN](./study/study_double_dqn.py)
  ```
  tested_env : CartPole

  Use the online network to find the index of the best action

  The only difference from DQN is 

      R + gamma * Q_Target(s', max_a'(Q_Online(s', a')))

  ```

  [Dueling DQN](./study/study_dueling_dqn.py)

  ```
  tested_env : CartPole

  
  Idea from 'Decoupling the action-indepent value of state and Q-value may lead to more robust learning'
  
  Q-Value can be split into state value and action advantage part
  Q(s, a) = V(s) + A(s, a)

  New network architure from this.
  ```

### TODO
- [x] Q-Learning
- [x] DQN
- [x] Double DQN
- [x] Dueling DQN
