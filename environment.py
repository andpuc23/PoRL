import gym
from gym import spaces
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from numpy.random import default_rng

class Environment(gym.env):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.max_capacity = 50 #kWh
        self.efficiency = 0.9
        self.power = 25 #kW
        self.morning_capacity = 20 #kWh
        self.rng = default_rng()
        self.battery_valuation = 1000
    
        #Define action space
        # 0-24 sell
        # 25 nothing
        # 26-51 buy
        self.action_space = spaces.Discrete(51)
        
        #Define observation space
        # battery, hour, price, availability
        
        self.observation_space = spaces.Dict({
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'price': spaces.Box(low=self.min_price, high=self.max_price),
            'availability': spaces.Discrete(2)  # 0: available, 1: unavailable
        })
        
        self.state = {
            'battery': self.morning_capacity,
            'hour': 0,
            'price': data.loc[data['day'] == self.t, 'price'].values,
            'availability': random.randint(0,1)
        }
    
    def reset(self):
        # Reset the environment to the initial state
        self.state = {
            'battery': self.morning_capacity,
            'hour': 8,
            'price': data.loc[data['day'] == self.t, 'price'].values,
            'availability': random.randint(0,1)
        }
        
        return self.state, reward, done  
    
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
    
    def render(self):
        pass
    
    def close(self):
        pass
