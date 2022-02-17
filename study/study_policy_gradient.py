import numpy as np
import torch
import torch.nn as nn
import gc
from torch.optim import SGD, Adam, RMSprop
from itertools import count
import time


##########################
# HYPERPARAMETERS
ENV_NAME = 'CartPole-v0'
NUM_EPISODES = 1000 
RENDER = False
E = 0.1 # e-greedy
LAMBDA = .99
LEARNING_RATE = 0.001
##########################

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


#=================================================================NETWORK

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


def update_network(target, online):
    """
    update the online network into target network
    """
    target.load_state_dict(online.state_dict())


def preprocess_state(state): 
    """
    convert into the torch
    """
    state = torch.tensor(state, device=device, dtype=torch.float32)
    return state


def select_action(model, state):
    state = preprocess_state(state)
    with torch.no_grad(): 
        probs = model(state).cpu().detach().numpy()
        dist = torch.distributions.Categorical(logits=probs)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(probs.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy



    
if __name__ == '__main__':
    import gym
    train = True
    test = True

    env = gym.make(ENV_NAME)

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    Model = DemoNetwork
    Optimizer = RMSprop
    

    step_count = 0

    model = Model(n_state, n_action).to(device)

    if train:
        print('Training {}'.format(ENV_NAME))

        online_model.train()
        target_model.eval()

        ##
        optimizer = Optimizer(online_model.parameters(), lr=LEARNING_RATE)

        start_time = time.time()

        for i in range(1, NUM_EPISODES+1):
            state, is_done = env.reset(), False
            episode_reward = 0
            if RENDER: env.render()

            ## BEGIN ::: Train Model
            ## each step, experience
            while True:
                if is_done:
                    # terminate the episode
                    break

                ## END::Update
            ## END ::: 

            # garbage collect
            gc.collect()

            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f} | elapsed time: {:.3f}'.format(i, NUM_EPISODES, episode_reward, time.time() - start_time))
            pass


    if test:
        print('')
        print('')
        print('')
        print('Testing {}'.format(ENV_NAME))
        for i in range(1, NUM_EPISODES+1):
            state = env.reset()
            episode_reward = 0
            if RENDER: env.render()

            while True:
                action = select_action(online_model, state, 0)
                # get new state and reward from environment
                new_state, reward, is_done, _ = env.step(action)

                episode_reward += reward
                state = new_state

                if is_done:
                    break

            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f}'.format(i, NUM_EPISODES, episode_reward))
            pass


