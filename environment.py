import gymnasium as gym
from gym import spaces
import numpy as np
# import time
# import matplotlib.pyplot as plt
# import time
import random
from numpy.random import default_rng

class Environment(gym.Env):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.max_capacity = 50 #kWh
        self.efficiency = 0.9
        self.rng = default_rng()
        self.battery_valuation = 1000
        
        self.power = 25 #kW
        self.morning_capacity = 20 #kWh
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
            'price': spaces.Box(low=0, high=self.max_price, shape=(1,)),
            'availability': spaces.Discrete(2)  # 0: available, 1: unavailable
        })
        self.state = dict()
        self.reset()


    def reset(self):
        """
        Reset the environment to the initial state
        Returns the initial state
        """
        self.__index = 0
        self.t = self.data.iloc[self.__index]
        

        self.state['hour'] = self.t.Hour
        self.state['battery'] = self.morning_capacity
        self.state['price'] = self.data.loc[self.data['Date'] == 0]['Price'].values
        self.state['availability'] = random.randint(0,1)
        
        return self.state
    
    def step(self, action):
        # value_battery = self.state['battery']*self.battery_valuation
        
        if action < 25: #sell
            reward = -(action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        elif action == 25:
            reward = 0
        else: # buy
            reward = (action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
        
        self.__index +=1 
        self.t = self.data.iloc[self.__index]
        self.state['hour'] = self.t.hour
        
        self.state['battery'] += action*self.efficiency
        self.state['price'] = self.data.loc[self.data['Date'] == self.date, 'Price'].values,
        self.state['availability'] = random.randint(0,1)
        
        return self.state, reward
