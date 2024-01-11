from abc import ABC


class Model(ABC):
    def train(self, data):
        """
        trains the model inside
        input: pd.DataFrame of 4 columns: ['Price', 'Summer_delta', 'Weekday', 'Hour']
        """
        pass

    def predict(self, data)->int:
        """
        applies the model to predict the data
        data: pd.DataFrame -- previous days to predict the action on
        Here we expect some previous observations to make a decision on 

        returns int encoding a decision
        25 for idle
        0-24 for sell
        26-50 for buy
        """
        pass