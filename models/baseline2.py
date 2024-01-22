from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel2(Model):
    def __init__(self, data:pd.DataFrame):      
        pass

    def predict(self, state):
        last_price = state['price']
        action = 25
        if state['battery'] < 20 and state['hour']==7:
            # action will at least be to charge up to 20 and possibly the action determined above
            action = max([action, 25+(20-state['battery'])])
        
        return action
    
    def train(self, data):
        pass
