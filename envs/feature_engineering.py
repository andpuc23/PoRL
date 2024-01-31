from envs.train_env import Electric_Car 
from envs.TestEnv import Electric_Car as Electric_Car_Test
import numpy as np
from collections import deque

'''
- day of week (int)
- month (int)
- weekend (binary)
- season (int)

- moving average (over 1 week)
- lagged price (yesterday same time)
- lagged price (a week ago)
- trend (difference with previous price)
- trend (diff with price yesterday same time)

- min/max price for a week
- std for a week
'''



class DataHelper:
    def __init__(self):
        self.previous_prices = deque(maxlen=7*24)

    def get_size(self):
        return 21


    def _get_season_from_month(self, month:int)->tuple:
        '''season_winter, season_spring, season_summer, season_fall'''
        if month <=2 or month == 12:
            return 1,0,0,0
        elif 3 <=month <=5:
            return 0,1,0,0
        elif 6<=month<=8:
            return 0,0,1,0
        else:
            return 0,0,0,1


    def _get_weekend(self, day_of_week)->bool:
        return day_of_week >= 5


    def process_data(self, observation):
        """
        returns new observation:
        battery, price, hour, day_of_week, day_of_year, month, year, car_is_available,
        season_winter, season_spring, season_summer, season_fall (1,0,0,0 for winter),
        weekend flag, average price for week, yesterday's price, week ago price,
        trend for yesterday, trend for week ago, week's min, week's max, week's std
        """
        battery, price, hour, day_of_week, day_of_year, month, year, car_is_available = observation
        engineered_observation = observation
        self.previous_prices.append(price)
        
        engineered_observation += self._get_season_from_month(month)
        
        engineered_observation.append(int(self._get_weekend(day_of_week)))

        average = np.average(self.previous_prices)
        engineered_observation.append(average)

        if len(self.previous_prices) > 24:
            yesterdays_price = self.previous_prices[-24]
        else:
            yesterdays_price = 0
        engineered_observation.append(yesterdays_price)
        
        if len(self.previous_prices) == 24*7:
            weeks_price = self.previous_prices[0]
        else:
            weeks_price = 0
        engineered_observation.append(weeks_price)

        engineered_observation.append(price - yesterdays_price)
        engineered_observation.append(price - weeks_price)

        engineered_observation.append(np.min(self.previous_prices))
        engineered_observation.append(np.max(self.previous_prices))
        engineered_observation.append(np.std(self.previous_prices))

        return np.array(engineered_observation)
