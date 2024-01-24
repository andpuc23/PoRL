# from ..envs.test_env import Electric_Car
from collections import deque
import torch
import random
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.abspath('..'))

class ExperienceReplay:
    
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed = 123):
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-200.0], maxlen = 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print('Please wait, the experience replay buffer will be filled with random transitions')
                
        obs, _ = self.env.reset()
        for _ in range(self.min_replay_size):
            
            action = (np.random.random()-.5)*2
            new_obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if done:
                obs, _ = env.reset()
        
        print('Initialization with random transitions is done!')
      
          
    def add_data(self, data): 
        self.replay_buffer.append(data)
            
    def sample(self, batch_size):
        
        transitions = random.sample(self.replay_buffer, batch_size)

        #Solution
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        #PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu / GPU)
        observations_t = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t
    
    def add_reward(self, reward):
        
        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''
        
        self.reward_buffer.append(reward)
        

class DQN(nn.Module):
    
    def __init__(self, env, learning_rate):
        
        super(DQN,self).__init__()
        input_features = env.observation_space.shape[0]
        
        self.dense1 = nn.Linear(in_features = input_features, out_features = 256)
        self.dense2 = nn.Linear(in_features = 256, out_features = 128)
        self.dense3 = nn.Linear(in_features = 128, out_features = 64)
        self.dense4 = nn.Linear(in_features = 64, out_features = 32)
        self.dense5 = nn.Linear(in_features = 32, out_features=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = torch.tanh(self.dense4(x))
        x = self.dense5(x)
        x = F.softmax(x)
        
        return x
    


class DDQNAgent:
    
    def __init__(self, data_path, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
        self.env_name = data_path
        # self.env = Electric_Car(path_to_test_data=data_path)
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        # self.replay_memory = ExperienceReplay(self.env, self.buffer_size, seed = seed)
        # self.online_network = DQN(self.env, self.learning_rate).to(self.device)
        
        # self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        # self.target_network.load_state_dict(self.online_network.state_dict())
        

    def init_envs(self, env):
        self.env = env
        self.replay_memory =  ExperienceReplay(env, self.buffer_size)
        
        self.online_network = DQN(env, self.learning_rate).to(self.device)        
        self.target_network = DQN(env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def choose_action(self, step, observation, greedy = False):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
    
        random_sample = random.random()
    
        if (random_sample <= epsilon) and not greedy:
            action = (np.random.random()-0.5)*2
            print()        
        else:
            #Greedy action
            obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            q_values = self.online_network(obs_t.unsqueeze(0))
        
            max_q_index = torch.argmax(q_values, dim = 1)[0]
            action = max_q_index.detach().item()
        
        return action, epsilon
    
    
    def return_q_value(self, observation):
        
        obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
        q_values = self.online_network(obs_t.unsqueeze(0))
        
        return torch.max(q_values).item()
        
    def learn(self, batch_size):
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        target_q_values = self.target_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values
        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = F.smooth_l1_loss(action_q_values, targets)
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    

    def load(self, path):
        self.target_network.load_state_dict(torch.load(path))

    def play_game(self, env):
        rewards = []
        actions = []
        done = False
        step = 0
        state, _ = env.reset()
        while not done:
        
            action = self.choose_action(step, state, True)[0]
            next_state, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            state = next_state
            rewards.append(rew)
            actions.append(action)
            step += 1 
        env.close()
        return rewards, actions
    


def training_loop(env, agent, max_episodes, target_ = False, seed=42, batch_size=32):
    
    obs, _ = env.reset()
    average_reward_list = [0]
    episode_reward = 0.0
    
    for step in range(max_episodes):
        
        action, epsilon = agent.choose_action(step, obs)
       
        new_obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated        
        transition = (obs, action, rew, done, new_obs)
        agent.replay_memory.add_data(transition)
        obs = new_obs
    
        episode_reward += rew
    
        if done:
        
            obs, _ = env.reset()
            agent.replay_memory.add_reward(episode_reward)
            episode_reward = 0.0

        #Learn

        agent.learn(batch_size)
                
        if step % 100 == 0:
            average_reward_list.append(np.mean(agent.replay_memory.reward_buffer))
        
        #Update target network, do not bother about it now!
        if target_:
            
            #Set the target_update_frequency
            target_update_frequency = 250
            if step % target_update_frequency == 0:
                agent.update_target_network()
    
        #Print some output
        if (step+1) % 1000 == 0:
            print(20*'--')
            print('Step', step)
            print('Epsilon', epsilon)
            print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
            print()

    return average_reward_list