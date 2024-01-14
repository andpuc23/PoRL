import gymnasium as gym
from gym import spaces
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from numpy.random import default_rng

class Environment(gym.Env):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.max_capacity = 50 #kWh
        self.efficiency = 0.9
        self.power = 25 #kW
        self.morning_capacity = 20 #kWh
        self.rng = default_rng()
        self.battery_valuation = 20
        self.min_price = 0
        self.max_price = 10000
    
        #Define action space
        # 0-24 kWh sell
        # 25 nothing
        # 26-50 kWh buy
        self.action_space = spaces.Discrete(51)
        
        #Define observation space
        # battery, hour, price, availability
        
        self.observation_space = spaces.Dict({
            't': data.length,
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'price': spaces.Box(low=self.min_price, high=self.max_price, dtype=np.float32),
            'availability': spaces.Discrete(2),  # 0: available, 1: unavailable
            'distance_summer': spaces.Discrete(6)
        })
        
        self.state = {
            't': 1,
            'battery': self.morning_capacity,
            'hour': 1,
            'price': data.loc[1, 'price'],
            'availability': random.randint(0,1),
            'distance_summer': data.loc[1, 'Summer_delta']
        }
    
    def reset(self, data):
        # Reset the environment to the initial state
        self.state = {
            't': 8,
            'battery': self.morning_capacity,
            'hour': 8,
            'price': data.loc[8, 'price'],
            'availability': random.randint(0,1),
            'distance_summer': data.loc[8, 'Summer_delta']
        }
        
        return self.state, 0
    
    def step(self, action, data):
        if action < 25: #sell
            reward = -(action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        elif action > 25: #buy
            reward = 2*(action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        
        self.state['battery'] += (action-25)*self.efficiency

        if self.state['hour'] == 24:
            self.state['hour'] = 0
        else:
            self.state['hour'] +=1

        self.state['t']  += 1

        self.state['price'] = data.loc[self.state['t'], 'price']

        self.state['availablity'] = random.randint(0,1)

        self.state['distance_summer'] = data.loc[self.state['t'], 'Summer_delta']

        

        return self.state, reward
