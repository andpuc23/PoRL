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
        self.battery_valuation = 1000
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
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'price': spaces.Box(low=self.min_price, high=self.max_price, dtype=np.float32),
            'availability': spaces.Discrete(2)  # 0: available, 1: unavailable
        })
        
        self.state = {
            'battery': self.morning_capacity,
            'hour': 0,
            'price': data.loc[data['day'] == 0, 'price'].values,
            'availability': random.randint(0,1)
        }
    
    def reset(self):
        # Reset the environment to the initial state
        self.state = {
            'battery': self.morning_capacity,
            'hour': 8,
            'price': data.loc[data['day'] == 0, 'price'].values,
            'availability': random.randint(0,1)
        }
        
        return self.state, 0
    
    def step(self, action):
        value_battery = self.state['battery']*self.battery_valuation
        
        if action < 25: #sell
            reward = -(action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        elif action > 26: #buy
            reward = (action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        
        self.state['battery'] += action*self.efficiency            
        self.state['hour'] += 1
        self.state['price'] = data.loc[data['Date'] == self.date, 'price'].values,
        self.state['availability'] = random.randint(0,1)
        
        return self.state, reward
