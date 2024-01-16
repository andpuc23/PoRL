import gymnasium as gym
from gym import spaces
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random

class Environment(gym.Env):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.max_capacity = 50 #kWh
        self.efficiency = 0.9
        self.power = 25 #kW
        self.morning_capacity = 20 #kWh
        self.battery_valuation = 10
        self.min_price = 0
        self.max_price = 10000
    
        #Define action space
        # 0-24 kWh sell, 0 is removing 25 from battery, so selling 25*0.9
        # 25 nothing
        # 26-50 kWh buy, 50 is adding 25 to the battery, so buying 25/0.9
        self.action_space = spaces.Discrete(51)
        
        #Define observation space
        # battery, hour, price, availability
        
        self.observation_space = spaces.Dict({
            't': spaces.Discrete(len(data)),
            'battery': spaces.Box(low=0, high=self.max_capacity, shape=(1,), dtype=np.float32),
            'hour': spaces.Discrete(24),
            'price': spaces.Box(low=self.min_price, high=self.max_price, shape=(1,), dtype=np.float32),
            'availability': spaces.Discrete(2),  # 0: unavailable, 1: available
            'distance_summer': spaces.Discrete(6) # distance in months
        })
        
        self.state = {
            't': 0,
            'battery': self.morning_capacity,
            'hour': 0,
            'price': data.iloc[0]['Price'],
            'availability': data.iloc[0]['Available'],
            'distance_summer': data.iloc[0]['Summer_delta']
        }
    
    def reset(self, data):
        # Reset the environment to the initial state
        self.state = {
            't': 0, #not the same as 'hour'
            'battery': self.morning_capacity,
            'hour': 0,
            'price': data.iloc[0]['Price'], 
            'availability': data.iloc[0]['Available'],
            'distance_summer': data.iloc[0]['Summer_delta']
        }
        
        return self.state, 0
    
    def step(self, action, data):
        if action < 25: #sell
            # action is netto removal from battery, you sell 22.5 and remove 25 from battery
            # i removed the minus, someone should check
            reward = (action-25)*self.efficiency*(self.battery_valuation - self.state['price'])
            self.state['battery'] += (action-25)
            
        elif action > 25: #buy
            #action is netto addition to battery, you buy 27.77 and get 25 in battery
            
            reward = 2*(action-25)/self.efficiency*(self.battery_valuation - self.state['price'])
            self.state['battery'] += (action-25)
            
        else: #do nothing
            reward = 0
        
        if self.state['hour'] == 23:
            self.state['hour'] = 0
        else:
            self.state['hour'] +=1
            
        self.state['t'] += 1
        self.state['price'] = data.iloc[self.state['t']]['Price']
        self.state['availability'] = data.iloc[self.state['t']]['Available']
        self.state['distance_summer'] = data.iloc[self.state['t']]['Summer_delta']

        return self.state, reward
