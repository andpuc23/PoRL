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
        self.min_price = 0
        self.max_price = 10000 
    
        #Define action space
        # Do nothing = 0
        # Buy = 1
        # Sell = 2
        self.action_space = spaces.Discrete(3)
        
        #Define observation space
        # battery, hour, date, price, availability
        
        self.observation_space = spaces.Dict({
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'date': spaces.Disctrete(365),
            'price': spaces.Box(low=self.min_price, high=self.max_price),
            'availability': spaces.Discrete(2)  # 0: available, 1: unavailable
        })
        
        self.state = {
            'battery': self.morning_capacity,
            'hour': 0,
            'date': 0,
            'price': random.uniform(self.min_price, self_max_price),
            'availability': random.randint(0,1)
        }
    
    def reset(self):
        # Reset the environment to the initial state
        self.state = {
            'battery': self.morning_capacity,
            'hour': 0,
            'date': 0,
            'price': random.uniform(self.min_price, self_max_price),
            'availability': random.randint(0,1)
        }
        
        return self.state, reward, done  
    
    def step(self, action)
        if action == 0:
            self.state['battery'] += self.power*self.efficiency

        elif action == 1:
            self.state['battery'] -= self.power*self.efficiency
            
        else:
            

        self.state['hour'] += 1
        self.state['date'] 
        self.state['price'] = random.uniform(self.min_price, self_max_price),
            'availability': random.randint(0,1)
        
        
        
        return self.state, reward
    
    def render(self):
        pass
    
    def close(self):
        pass
