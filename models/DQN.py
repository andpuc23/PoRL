from models.interface import Model

import random
import math
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, n_actions)
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.layers(x)
        

class DQNModel(Model):
    def __init__(self,
                 env,
                 batch_size:int=128,
                 gamma:float=.99,
                 eps_start:float=.9,
                 eps_end:float=.05,
                 eps_decay:float=1000, #higher is slower
                 tau:float=.005,
                 lr:float=1e-4,
                 replay_capasity:int=int(1e5)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.n_actions = self.env.action_space.n
    
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        observation, _ = self.env.reset()
        n_observations = len(observation)

        self.policy_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(replay_capasity)

        self.steps_done = 0


    def __select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end)*\
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def _train_one_epoch(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train(self, num_episodes=100):
        if num_episodes == 100:
            if  torch.cuda.is_available():
                num_episodes = 1000
            else:
                num_episodes = 100
        print('training on', 'cuda' if torch.cuda.is_available() else 'cpu')

        for i_episode in tqdm(range(num_episodes)):
            
            state, info = self.env.reset()
            state = list(state.values())
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in tqdm(count()):
                action = self.__select_action(state)
                observation, reward, done = self.env.step(action.item())
                observation = list(observation.values())
                reward = torch.tensor([reward], device=self.device)
            
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self._train_one_epoch()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break


        print('Done training')


    def save_model(self, path:str):
        torch.save(self.policy_net.state_dict(), path)

    
    def load_model(self, path:str):
        self.policy_net.load_state_dict(torch.load(path))


    def predict(self, state) -> int:
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)