import gymnasium as gym
from gym import spaces
import numpy as np
# import time
# import matplotlib.pyplot as plt
# import time
# import random
# from numpy.random import default_rng

class Environment(gym.Env):
    def __init__(self, data):
        super().__init__()
        
        self.data = data
        self.max_capacity = 50 #kWh
        self.efficiency = 0.9
        self.power = 25 #kW
        self.morning_capacity = 20 #kWh
        self.battery_valuation = 40
        self.min_price = 0
        self.max_price = 10000
    
        #Define action space
        # 0-24 sell
        # 25 nothing
        # 26-50 buy
        self.action_space = spaces.Discrete(51)
        
        #Define observation space
        # battery, hour, price, availability
        
        self.observation_space = spaces.Dict({
            # 't': spaces.Discrete(len(data)),
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'price': spaces.Box(low=self.min_price, high=self.max_price, dtype=np.float32, shape=(1,)),
            'availability': spaces.Discrete(2),  # 0: unavailable, 1: available
            'distance_summer': spaces.Discrete(7) # distance in months
        })
        
        self.state = {
            't': 0,
            'battery': self.morning_capacity,
            'hour': 0,
            'price': data.iloc[0]['Price'],
            'availability': data.iloc[0]['Available'],
            'distance_summer': data.iloc[0]['Summer_delta']
        }
    
    def reset(self):
        # Reset the environment to the initial state
        self.state = {
            't': 0,
            'battery': self.morning_capacity,
            'hour': 0,
            'price': self.data.iloc[0]['Price'],
            'availability': self.data.iloc[0]['Available'],
            'distance_summer': self.data.iloc[0]['Summer_delta']
        }
        
        observable_state = self.state.copy()
        del observable_state['t']
        
        return observable_state, 0
    
    def step(self, action):
        if action < 25 and self.state['availability'] == 1: #sell
            #reward = (action-25)*(self.battery_valuation - self.state['price'])
            reward = (25-action)/self.efficiency*self.state['price']
            self.state['battery'] += (action-25)/self.efficiency

        elif action > 25 and self.state['availability'] == 1: #buy
            #reward = 2*(action-25)/self.efficiency*(self.battery_valuation - self.state['price'])
            reward = 2*(25-action)/self.efficiency*self.state['price']
            self.state['battery'] += (action-25)

        else: #do nothing
            reward = 0

        if self.state['battery'] < 20 and self.state['hour']==7:
            # action will at least be to charge up to 20 and possibly the action determined above
            action = 25+(20-self.state['battery'])
            reward += 2*(25-action)/self.efficiency*self.state['price']

        
        
        if self.state['availability']==0:
            self.state['battery']-=2
        
        if self.state['hour'] == 23:
            self.state['hour'] = 0
        else:
            self.state['hour'] +=1

        self.state['t']  += 1
        self.state['price'] = self.data.iloc[self.state['t']]['Price']
        self.state['availability'] = self.data.iloc[self.state['t']]['Available']
        self.state['distance_summer'] = self.data.iloc[self.state['t']]['Summer_delta']

        observable_state = self.state.copy()
        del observable_state['t']
        finished = self.state['t'] >= self.data.shape[0]-1

        return observable_state, reward, finished
