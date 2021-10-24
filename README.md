# Study for Reinforcement Learning with PyTorch

* **Q-Learning** 

  [Python test_code](./study/study_qlearning.py)

  ```
  tested_env : FrozenLake

  Update equation
    Q(S, A) <- Q(S, A) + alpha * (R + lambda * max_a( Q(next_S, :) ) - Q(S, A))
  
  ```

* **Deep Q-Network (DQN)**

  [Python test_code](./study/study_qlearning.py)

  ```
  tested_env : CartPole

  Main step for algorithms

    1. Randomly sampled a mini-batch from replay buffer.

    2. calculate the off-policy TD target with target network.
      R + gamma * max_a'(Q_Target(s',a'))

    3. Fit action-value function Q using MSE and RMSProp
  ```



