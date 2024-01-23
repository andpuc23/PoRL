from models.interface import Model
import numpy as np
import pandas as pd

class BaselineModel2(Model):
    def __init__(self, data:pd.DataFrame):      
        pass

    def predict(self, env):
        return 0
    
    def train(self, data):
        pass
