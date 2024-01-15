from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel(Model):
    def __init__(self, data:pd.DataFrame):
        self.lower_thres = np.quantile(data.Price.values, 0.25)
        self.higher_thres = np.quantile(data.Price.values, 0.75)

    def predict(self, data, state):
        last_row = data.iloc[-1]
        last_price = last_row.Price
        if self.lower_thres < last_price < self.higher_thres:
            action = 25 # do nothing
        elif self.lower_thres > last_price:
            action = 50 # buy 25 cheap
        else:
            action = 0 # sell 25 expensive  

        return action
    
    
    def train(self, data):
        pass
