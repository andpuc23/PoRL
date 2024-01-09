from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel(Model):
    def __init__(self, data:pd.DataFrame):
        super().__init__(self)
        data_cols = sorted([c for c in data.columns if c.startswith('Hour')])
        assert len(data_cols) == 24, f"number of hours in a day should be 24, got {len(data_cols)}"

        lower_thres = []
        higher_thres = []
        for col in data_cols:
            lower_thres.append(np.quantile(data[col], 0.25))
            higher_thres.append(np.quantile(data[col], 0.75))
    

    def predict(self, data):
        last_row = data.iloc[-1]
        last_price = last_row.Price
        if self.lower_thres < last_price < self.higher_thres:
            action = 0 # do nothing
        elif self.lower_thres > last_price:
            action = 1 # buy cheap
        else:
            action = -1 # sell expensive

        return action
    
    
    def train(self, data):
        pass