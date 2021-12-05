import numpy as np
import torch
import torch.nn as nn
import gc
from torch.optim import SGD, Adam, RMSprop
from itertools import count
import time



#####
##### ::::::::::::::::::

class Opt:
    def __init__(self, env):
        self.env = env
        self.random_seed = None


class BaseAgent:
    def __init__(self, opt):
        self.opt = opt
        self.env = opt.env
        self.random_seed = opt.random_seed

        if self.random_seed is not None:
            self.set_random_seed(opt.random_seed)

    def set_random_seed(self, seed):
        torch.manual_seed(opt.random_seed)
        np.random.seed(opt.random_seed)


    def reset(self):
        if self.random_seed is not None:
            self.set_random_seed(self.random_seed)
        self.state = self.env.reset()






class DQNModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_dueling = False
        try:
            self.is_dueling = self.model.v is not None
        except:
            pass

    def forward(self, state):
        x = state
        a = self.model(x)
        
        if not is_dueling:
            return a

        # if model is for dueling
        v = self.model.v.expand_as(a)
        y = v + a - a.mean(1, keepdim=True).expand_as(a)
        return y


class ReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self._offset = 0
        # 5 : (state, action, reward, new_state, is_done)
        self.buffer = np.zeros(shape=(capacity, 5), dtype=np.ndarray)

    def add(self, experience:tuple):
        """
        experience is tuple of ndarray
        """
        self.buffer[self._offset] = experience
        self._offset = (self._offset+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, device):
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

##### ::::::::::::::::::


#####
##### ::::::::::::::::::INTERFACE
class BaseActionStrategy:
    def reset(self):
        pass
    def select_action(self, model, state):
        pass

class BaseReplayBuffer:
    def __init__(self):
        self.size = 0
    def __len__(self):
        return self.size
    def reset(self):
        pass
    def add(self, experience:tuple):
        pass
    def sample(self, batch_size, device):
        pass
##### ::::::::::::::::::



#####
##### ::::::::::::::::::ACTION STRATEGY
class GreedyActionStrategy(BaseActionStrategy):
    def reset(self): pass

    def select_action(self, model, state):
        with torch.no_grad(): 
            pred = model(state).cpu().detach().numpy()
            action = np.argmax(pred) 
        return action

class BaseEGreedyStrategy(BaseActionStrategy):
    def __init__(self):
        self.step = 0

    def reset(self):
        self.step = 0

    def _epsilon(self):
        return 0.0

    def select_action(self, model, state):
        with torch.no_grad(): 
            pred = model(state).cpu().detach().numpy()
            action = np.argmax(pred) if np.random.rand() > self._epsilon() else np.random.randint(len(pred))
        return action


class NormalEGreedyStrategy(BaseEGreedyStrategy):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def _epsilon(self):
        return self.epsilon

class LinearEGreedyStrategy(BaseEGreedyStrategy):
    def __init__(self, start_epsilon=1., end_epsilon=0.1, decay_steps=2000):
        super().__init__()
        self.start_e = start_epsilon
        self.end_e = end_epsilon
        self.decay_steps = decay_steps

    def _epsilon(self):
        i = self.step 
        self.step+=1
        ratio = 1 - (i/self.decay_steps)
        e = (self.start_e-self.end_e)*ratio + self.end_e
        return np.clip(e, self.end_e, self.start_e)


class ExponentialDecayEGreedyStrategy(BaseEGreedyStrategy):
    def __init__(self, start_epsilon=1., end_epsilon=0.1, decay_steps=2000):
        super().__init__()
        self.start_e = start_epsilon
        self.end_e = end_epsilon
        self.decay_steps = decay_steps

    def _epsilon(self):
        i = self.step 
        self.step+=1
        exp_e = self.start_e* np.power(0.005, (i / self.decay_steps))
        return exp_e * (self.start_e- self.end_e) + self.end_e
##### ::::::::::::::::::




#####
##### ::::::::::::::::::MODEL
class DQN:
    """
    model_builder_fn : callable function. model should have the 'v' property. This will determine whether the model is Dueling DQN or not.
    """
    def __init__(self, model_builder_fn):
        self.online = DQNModelWrapper(model_builder_fn())
        self.target = DQNModelWrapper(model_builder_fn())
        pass

    def update_target_network(self):
        self.target.load_state_dict(self.online.state_dict())
        pass


    def learn(self, env, buffer):
        # self.buffer = replay_buff_obj
                 # optimizer_fn, 
                 # optimizer_lr, 
        state = env.reset()

        pass


    def test(self, device):
        pass


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.b = self.b.to(*args, **kwargs) 
        return self



##### ::::::::::::::::::










if __name__ == '__main__':

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    dqn = DQN(









