import gym
from itertools import count
import numpy as np


## HYPERPARAMETERS
LAMBDA = 0.9
ALPHA = 0.1
IS_SLIPPERY = False
##

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.reset()
        self.n_action = env.action_space.n
        self.n_state = env.observation_space.n
        self.q_table = np.zeros((self.n_state, self.n_action), dtype=float)
        pass

    def reset(self):
        self.state = self.env.reset()

    def random_step(self):
        action = self.env.action_space.sample()
        state = self.state
        new_state, reward, is_done, _ = self.env.step(action)

        # if done, then reset
        if is_done: self.reset()
        else: self.state = new_state
        return state, action, reward, new_state, is_done

    def greedy_step(self, state):
        action = np.argmax(self.q_table[state])
        return action

    def update(self, experience):
        """
        Q(S, A) <- Q(S, A) + alpha * (R + lambda * max (Q(next_S, Action)) - Q(S, A))
        """
        s, a, r, next_s, d = experience
        td_target =  r + LAMBDA*np.max(self.q_table[next_s, :])
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] = self.q_table[s,a] + ALPHA*td_error

    def play(self, episode:int, render=False):
        self.reset()
        if render: self.env.render()
        state = self.state
        total_reward = 0.0

        for i in count():
            a = agent.greedy_step(state)
            s, r, is_done, _ = env.step(a)
            if render: self.env.render()
            total_reward += r
            state = s

            if is_done:
                print("  Episode {}, total reward: {}".format(episode, total_reward))
                return total_reward, i



if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=IS_SLIPPERY)

    agent = QLearningAgent(env)

    for i in count():
        # Training
        agent.reset()
        while True:
            state, action, reward, new_state, is_done = agent.random_step()
            # udpate the q value
            agent.update((state, action, reward, new_state, is_done))
            if is_done:
                break

        # test the episode
        r = 0.0
        for _ in range(10):
            reward, step_count = agent.play(i+1, False)
            r += reward
        r /= 10
        if r > 0.9:
            _, count = agent.play(-1, True)
            print('Final Step count : {}'.format(count)) 
            
            break
            
            



    pass
