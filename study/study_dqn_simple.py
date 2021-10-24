import numpy as np
import torch
import torch.nn as nn
import gc
from torch.optim import SGD, Adam, RMSprop
from itertools import count
import time


##########################
# HYPERPARAMETERS
# ENV_NAME = 'FrozenLake-v1'
ENV_NAME = 'CartPole-v0'
NUM_EPISODES = 1000 
RENDER = False
E = 0.1 # e-greedy
LAMBDA = .99
LEARNING_RATE = 0.001
UPDATE_TARGET_STEP = 5
BATCH_SIZE = 64
BUFFER_CAPACITY = 1000
##########################

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


#=================================================================NETWORK
# Q-network -> input - 32 - output with relu
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

#=================================================================REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self._offset = 0
        # 5 : (state, action, reward, new_state, is_done)
        self.buffer = np.zeros(shape=(capacity, 5), dtype=np.ndarray)

    def __len__(self):
        return self.size

    def add(self, experience:tuple):
        """
        experience is tuple of ndarray
        """
        self.buffer[self._offset] = experience
        self._offset = (self._offset+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        """
        to torch.tensor
        """
        s, a, r, ss, d = self._sample(batch_size)
        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a).long().to(device)
        r = torch.from_numpy(r).float().to(device)
        ss = torch.from_numpy(ss).float().to(device)
        d = torch.from_numpy(d).float().to(device)
        return (s, a, r, ss, d)

    def _sample(self, batch_size):
        """
        get samples
        """
        idxes = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.buffer[idxes, 0]), np.vstack(self.buffer[idxes, 1]), np.vstack(self.buffer[idxes, 2]), np.vstack(self.buffer[idxes, 3]), np.vstack(self.buffer[idxes, 4])
        return experiences




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

def e_greedy_strategy(model, state, epsilon):
    state = preprocess_state(state)
    with torch.no_grad(): 
        pred = model(state).cpu().detach().numpy()
        action = np.argmax(pred) if np.random.rand() > epsilon else np.random.randint(len(pred))
        return action




    
if __name__ == '__main__':
    import gym

    train = False
    test = True


    env = gym.make(ENV_NAME)

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    Model = DemoNetwork
    Optimizer = RMSprop
    

    step_count = 0

    online_model = Model(n_state, n_action).to(device)
    target_model = Model(n_state, n_action).to(device)
    update_network(target_model, online_model)

    if train:
        print('Training {}'.format(ENV_NAME))

        online_model.train()
        target_model.eval()
        buffer = ReplayBuffer(BUFFER_CAPACITY)


        ##
        optimizer = Optimizer(online_model.parameters(), lr=LEARNING_RATE)

        start_time = time.time()

        for i in range(1, NUM_EPISODES+1):
            state = env.reset()
            episode_reward = 0
            if RENDER: env.render()


            ## BEGIN ::: Train Model
            ## each step, experience
            while True:
                ## BEGIN::Get Experience
                # choose action with e-greedy exploration.
                action = e_greedy_strategy(online_model, state, E)
                # get new state and reward from environment
                new_state, reward, is_done, _ = env.step(action)
                if RENDER: env.render()

                experience = (state, action, reward, new_state, is_done)
                buffer.add(experience)


                ## END::Get Experience
                if len(buffer) > BATCH_SIZE:
                    experiences = buffer.sample(BATCH_SIZE)

                    ## BEGIN::Optimize Model
                    with torch.no_grad():
                        states, actions, rewards, next_states, is_dones = experiences
                        max_a_q_value = target_model(preprocess_state(next_states)).detach().max(1)[0].unsqueeze(1)
                        target_q_value = rewards + (LAMBDA* max_a_q_value * (1 - is_dones))
                        # print(target_q_value)
                    q_value = online_model(preprocess_state(states)).gather(1, actions)
                    # td_error = q_value -  target_q_value
                    optimizer.zero_grad()
                    value_loss = nn.MSELoss()(q_value, target_q_value)
                    value_loss.backward()
                    optimizer.step()
                    ## END::Optimize Model

                    ## BEGIN::Update
                    step_count += 1
                    episode_reward += reward
                    state = new_state

                    if step_count % UPDATE_TARGET_STEP == 0:
                        update_network(target_model, online_model)

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
                action = e_greedy_strategy(online_model, state, 0)
                # get new state and reward from environment
                new_state, reward, is_done, _ = env.step(action)

                episode_reward += reward
                state = new_state

                if is_done:
                    break

            # if i%5 == 0:
            print('\tepisode: {}/{}  | total reward : {:.3f}'.format(i, NUM_EPISODES, episode_reward))
            pass


