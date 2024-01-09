from abc import ABC


class Model(ABC):
    def __init__(self):
        pass

    def train(self, data):
        """
        trains the model inside
        input: pd.DataFrame of 3 columns: date, price value and summer_delta - distance to July

        """
        pass

    def predict(self, data)->int:
        """
        applies the model to predict the data
        data: pd.DataFrame -- previous days to predict the action on
        Here we expect some previous observations to make a decision on 

        returns int encoding a decision
        0 for idle
        -1 for sell
        1 for buy
        """
        pass