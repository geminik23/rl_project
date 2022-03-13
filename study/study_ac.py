import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
import torch.nn.functional as F

##########################
# HYPERPARAMETERS
ENV_NAME = 'CartPole-v0'

NUM_EPISODES = 100000
NUM_TEST_EPISODES = 200

RENDER = False
LAMBDA = .99
LR_ACTOR = 0.01
LR_CRITIC = 0.01
##########################


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

def preprocess_state(state): 
    """
    convert into the torch
    """
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    return state

class Actor:
    def __init__(self, model_fn):
        self.model = model_fn()
        self.model.train()
        self.optim = Adam(self.model.parameters(), lr=LR_ACTOR)
        pass

    def select_action(self, state):
        state = preprocess_state(state)
        probs = self.model.forward(state)
        dist = torch.distributions.Categorical(logits=probs)
        action = dist.sample()
        entropy = dist.entropy().unsqueeze(-1)
        logpa = dist.log_prob(action).unsqueeze(-1)
        return action.item(), logpa, entropy

    def learn(self, state, td_error):
        # loss = tf.reduce_sum(tf.multiply(cross_entropy, rewards))
        _ , logpa, entropy =  self.select_action(state)
        loss = -(entropy*td_error).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

class Critic:
    def __init__(self, model_fn):
        self.model = model_fn()
        self.model.train()
        self.optim = Adam(self.model.parameters(), lr=LR_CRITIC)
        pass

    def learn(self, state, reward, next_state, is_done):
        v_1 = self.model(preprocess_state(next_state))
        v = self.model(preprocess_state(state))
        ## used td error as advantage 
        td_target = reward + (LAMBDA*v_1* (1 - is_done))
        td_error = td_target - v
        loss = td_error.pow(2).mul(0.5).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return td_error.detach()

def get_actor_model(n_state, n_action):
    model = nn.Sequential(
            nn.Linear(n_state, 32),
            nn.Tanh(),
            nn.Linear(32, n_action)
            )
    model.to(device)
    return model

def get_critic_model(n_state):
    model = nn.Sequential(
            nn.Linear(n_state, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
            )
    model.to(device)
    return model




import gym
import time

if __name__ == '__main__':
    train = True
    test = True

    env = gym.make(ENV_NAME)

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    actor = Actor(lambda : get_actor_model(n_state, n_action))
    critic = Critic(lambda: get_critic_model(n_state))

    if train:
        start_time = time.time()
        for i in range(1, NUM_EPISODES+1):
            state, is_done = env.reset(), False

            episode_total_reward = 0

            while True:
                action, logpa, entropy = actor.select_action(state)
                next_state, reward, is_done, _ = env.step(action)
                if RENDER: env.render()
                episode_total_reward += reward

                ## learn
                td_error = critic.learn(state, reward, next_state, is_done)
                actor.learn(state, td_error)

                if is_done:
                    break
                
                state = next_state
            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f} | elapsed time: {:.3f}'.format(i, NUM_EPISODES, episode_total_reward, time.time() - start_time))
            pass
        pass

    if test:
        pass


    
    pass
