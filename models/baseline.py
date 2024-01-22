from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel(Model):
    def __init__(self, env, Min, Max):      
        self.lower_thres = np.quantile(env.price_values, Min)
        self.higher_thres = np.quantile(env.price_values, Max)
        print(self.lower_thres)
        print(self.higher_thres)
        
    def predict(self, env):
        state = env.observation()
        if (env.hour == 24):
            current_day = env.day + 1
            current_hour = 1
        else:
            current_day = env.day
            current_hour = env.hour + 1


        last_price = env.price_values[current_day-1][current_hour-1]

        if self.lower_thres > last_price:
            action = 1 # buy max 25 cheap
        elif self.higher_thres < last_price:
            action = -1 
        else:
            action = 0 # do nothing

        return action
    
    def train(self, data):
        pass
