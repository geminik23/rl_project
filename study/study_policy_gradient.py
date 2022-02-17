import numpy as np
import torch
import torch.nn as nn
import gc
from torch.optim import SGD, Adam, RMSprop
import time


##########################
# HYPERPARAMETERS
ENV_NAME = 'CartPole-v0'
NUM_EPISODES = 1000
NUM_TEST_EPISODES = 200
RENDER = False

GAMMA = 0.99
LEARNING_RATE = 0.01
##########################

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


class DemoNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
                )

    def forward(self, state):
        x = state
        return self.model(x)

def preprocess_state(state): 
    """
    convert into the torch
    """
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    return state

def select_action(model, state):
    state = preprocess_state(state)
    probs = model.forward(state)
    dist = torch.distributions.Categorical(logits=probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action).unsqueeze(-1)
    

if __name__ == '__main__':
    import gym
    train = True
    test = True

    env = gym.make(ENV_NAME)

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    Model = DemoNetwork
    Optimizer = Adam
    model = Model(n_state, n_action).to(device)

    if train:
        print('Training {}'.format(ENV_NAME))
        model.train()

        optimizer = Optimizer(model.parameters(), lr=LEARNING_RATE)
        ##
        start_time = time.time()
        for i in range(1, NUM_EPISODES+1):
            state, is_done = env.reset(), False

            episode_total_reward = 0
            episode_rewards = []
            episode_logpa = []

            ## each step, experience
            while True:
                action, logpa = select_action(model, state)
                next_state, reward, is_done, _ = env.step(action)
                if RENDER: env.render()

                episode_rewards.append(reward)
                episode_logpa.append(logpa)
                episode_total_reward += reward

                state = next_state

                if is_done:
                    # terminate the episode
                    break

            ## BEGIN ::: Train Model
            
            # discounts factores
            len_t = len(episode_rewards)

            optimizer.zero_grad()
            # discounts * rewards
            discounts = np.logspace(0, len_t, num=len_t, base=GAMMA, endpoint=False)
            returns = np.array([np.sum(discounts[:len_t-i]*episode_rewards[i:]) for i in range(len_t)])

            discounts = torch.FloatTensor(discounts).unsqueeze(1).to(device)
            returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
            logpa = torch.cat(episode_logpa).to(device)

            ploss = -(discounts*returns*logpa).mean()
            optimizer.zero_grad()
            ploss.backward()
            optimizer.step()
            ## END ::: 
            gc.collect()

            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f} | elapsed time: {:.3f}'.format(i, NUM_EPISODES, episode_total_reward, time.time() - start_time))
            pass


    if test:
        print('')
        print('')
        print('')
        print('Testing {}'.format(ENV_NAME))
        model.eval()
        for i in range(1, NUM_TEST_EPISODES+1):
            state = env.reset()
            episode_reward = 0
            if RENDER: env.render()

            while True:
                action, _ = select_action(model, state)

                # get new state and reward from environment
                new_state, reward, is_done, _ = env.step(action)

                episode_reward += reward
                state = new_state

                if is_done:
                    break

            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f}'.format(i, NUM_TEST_EPISODES, episode_reward))
            pass