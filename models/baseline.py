from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel(Model):
    def __init__(self, data:pd.DataFrame):      
        self.lower_thres = np.quantile(data.Price.values, 0.25)
        self.higher_thres = np.quantile(data.Price.values, 0.75)
        
    def predict(self, data, state):
        last_row = data.iloc[-1]
        #last_price = last_row.Price
        t=state['t']
        last_price = data.iloc[t]['Price']
        
        if self.lower_thres > last_price:
            action = min([25 + (50-state['battery']), 50]) # buy max 25 cheap
        elif self.higher_thres < last_price:
            action = max([25-state['battery'], 0]) # sell max 25 expensive 
        else:
            action = 25 # do nothing
            
        if state['battery'] < 20 and state['hour']==7:
            # action will at least be to charge up to 20 and possibly the action determined above
            action = max([action, 25+(20-state['battery'])])
        
        return action
    
    def train(self, data):
        pass
