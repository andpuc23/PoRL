import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from train_env import Electric_Car 
import torch
import matplotlib.patches as mpatches
from envs.feature_engineering import DataHelper

class Agent():
    def __init__(self, data_path_train):
        
        '''
        Params:
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
         
        self.discount_rate = 0.95
        self.bin_size = [4, 6, 7, 3, 4, 4]  
        self.env = Electric_Car_Train(path_to_train_data=data_path_train)
        self.features_qtable = [0, 1, 2, 3, 5] 
        self.dh = DataHelper()
        
        # Discretize the action space and the state space of specified features
        self.action_space = np.linspace(-1, 1, self.bin_size[-1]-1)
        self.make_bins()
        
        
    def discretize_state(self, state):
        
        '''
        Params:
        state = state observation that needs to be discretized
        
        
        Returns:
        discretized state
        '''
        
        self.state = state
        digitized_state = []
        
        # Discretize the features of the state
        bins_idx = 0
        for i in range(len(self.state)):
            if i in self.features_qtable:
                digitized_state.append(np.digitize(self.state[i], self.bins[bins_idx])-1)
                bins_idx+=1
            else:
                digitized_state.append(int(self.state[i]))
        
        return np.array(digitized_state)  
    
    
    def make_bins():
        self.bins = []
 
        for i in range(len(self.features_qtable)):
        
            if self.features_qtable[i] == 0: # Battery 
                bins_feature = np.linspace(0, self.env.battery_capacity, self.bin_size[i])
                bins_feature[-1]+=0.1
                
            elif self.features_qtable[i]==1: # Price 
                bins_feature = np.linspace(0, np.percentile(self.env.price_values, 90), self.bin_size[i]-1)
            
            elif self.features_qtable[i]==2: # Hour
                bins_feature = np.linspace(1, 24, self.bin_size[i])
                bins_feature[-1]+=0.1
            
            elif self.features_qtable[i]==3: # Weekday or weekend
                bins_feature = np.array([0, 5, 7])
            
            elif self.features_qtable[i]==5: # Month
                bins_feature = np.linspace(1, 12, self.bin_size[i])
                bins_feature[-1]+=0.1
            
            self.bins.append(bins_feature) 
            
        return

    
    def create_Q_table(self):
        '''
        Returns:
        Q-table with zeros
        '''
        
        self.state_space = np.array(self.bin_size)-1
        self.Qtable = np.zeros(self.state_space)
        self.Qtable_updates = self.Qtable.copy()

    def train(self):
        
        '''
        Params:
        
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        
        '''
        
        simulations = 10
        self.epsilon = 0.05
        self.epsilon_decay = simulations
        self.learning_rate = 0.1
        self.epsilon_start = 1
        self.epsilon_end = 0.05
        self.sims_per_avg = simulations/10
        adaptive_epsilon = True
        adapting_learning_rate = False

        self.rewards = []
        self.average_rewards = []
        self.create_Q_table()
        
        
        if adapting_learning_rate:
            self.learning_rate = 1
        
        for i in range(simulations):
            if i % self.sims_per_avg == 0:
                print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            
            done = False
            
            # Reset and discretize state
            state = self.env.reset()[0]
            state = self.env.observation()
            state = self.discretize_state(state)
            
            total_rewards = 0
            
            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

            while not done:
                # Pick random or greeady action
                if np.random.uniform(0,1) > 1-self.epsilon:
                    action = np.random.choice(self.action_space)
                else: 
                    idx_action = np.argmax(self.Qtable[tuple(state[self.features_qtable])])
                    action = self.action_space[idx_action]
                
                #
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done =  terminated or truncated
                #next_state = dh.process_data(next_state)
                
                # Discretize the next state and the action
                next_state = self.discretize_state(next_state)
                idx_action = np.where(self.action_space==action)
               
                #Updating the Q-table
                Q_target = (reward + self.discount_rate*np.max(self.Qtable[tuple(next_state[self.features_qtable])]))
                delta = self.learning_rate * (Q_target - self.Qtable[tuple(state[self.features_qtable]) + (idx_action,)])
                self.Qtable[tuple(state[self.features_qtable]) + (idx_action,)] = self.Qtable[tuple(state[self.features_qtable]) + (idx_action,)] + delta
                self.Qtable_updates[tuple(state[self.features_qtable]) + (idx_action,)] += 1
                
                total_rewards += reward
                state = next_state
            
            if adapting_learning_rate:
                self.learning_rate = self.learning_rate/np.sqrt(i+1)
            
            self.rewards.append(total_rewards)

            if i % self.sims_per_avg == 0:
                self.average_rewards.append(np.mean(self.rewards))
                print('Average reward ', np.mean(self.rewards))
                self.rewards = []
            
        print('The simulation is done!')
        
        return
        
    def act(self, obs):
        obs = self.discretize_state(state)
            
        # Find optimal action
        idx_action = np.argmax(self.Qtable[tuple(state[self.features_qtable])])
        action = self.action_space[idx_action]

        return action

    def save_Qtable(self, file):
        np.save(file, self.Qtable)
        return 
    
    def load_Qtable(self, file):
        self.Qtable = np.load(file)
        return 

class Electric_Car_Train(gym.Env):
    def __init__(self, path_to_train_data=str):
        # Define a continuous action space, -1 to 1. (You can discretize this later!)
        self.continuous_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define the train data
        self.train_data = pd.read_excel(path_to_train_data)
        self.price_values = self.train_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.train_data['PRICES']
        self.state = np.empty(8)
        self.observation_space = self.state
        self.battery_valuation = self.battery_val() #list of mean price per month (0=jan, ...)
        
        # Battery characteristics
        self.battery_capacity = 50  # kWh
        self.max_power = 25 / 0.9  # kW
        self.charge_efficiency = 0.9  # -
        self.discharge_efficiency = 0.9  # -
        self.battery_level = self.battery_capacity / 2  # kWh (start at 50%)
        self.minimum_morning_level = 20  # kWh
        self.car_use_consumption = 20  # kWh

        # Time Tracking
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.car_is_available = True
        
    def battery_val(self):
        result = []
        for i in range(12):
            self.train_data['PRICES'] = pd.to_datetime(self.train_data['PRICES'])
            data_month = self.train_data[self.train_data['PRICES'].dt.month == i + 1] 
            month_data = data_month.iloc[:, 1:25].to_numpy()
            result.append(month_data.mean())
        return result


    def step(self, action):

        action = np.squeeze(action)  # Remove the extra dimension
        

        if action <-1 or action >1:
            raise ValueError('Action must be between -1 and 1')
        initial_action = action
        # Calculate if, at 7am and after the chosen action, the battery level will be below the minimum morning level:
        if self.hour == 7:
            if action > 0 and self.battery_level < self.minimum_morning_level:
                if self.battery_level + action * self.max_power * self.charge_efficiency < self.minimum_morning_level:  # If the chosen action will not charge the battery to 20kWh
                    action = (self.minimum_morning_level - self.battery_level) / (
                                self.max_power * self.charge_efficiency)  # Charge until 20kWh
            elif action < 0:
                if (self.battery_level + action * self.max_power) < self.minimum_morning_level:
                    if self.battery_level < self.minimum_morning_level:  # If the level was lower than 20kWh, charge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level) / (
                                    self.max_power * self.charge_efficiency)  # Charge until 20kWh
                    elif self.battery_level >= self.minimum_morning_level:  # If the level was higher than 20kWh, discharge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level) / (
                            self.max_power)  # Discharge until 20kWh
            elif action == 0:
                if self.battery_level < self.minimum_morning_level:
                    action = (self.minimum_morning_level - self.battery_level) / (
                                self.max_power * self.charge_efficiency)

        # There is a 50% chance that the car is unavailable from 8am to 6pm
        if self.hour == 8:
            self.car_is_available = np.random.choice([True, False])
            if not self.car_is_available:
                self.battery_level -= self.car_use_consumption
        if self.hour == 18:
            self.car_is_available = True
        if not self.car_is_available:
            action = 0

        # Calculate the costs and battery level when charging (action >0)
        if (action > 0) and (self.battery_level <= self.battery_capacity):
            if (self.battery_level + action * self.max_power * self.charge_efficiency) > self.battery_capacity:
                action = (self.battery_capacity - self.battery_level) / (self.max_power * self.charge_efficiency)
            charged_electricity_kW = action * self.max_power
            charged_electricity_costs = charged_electricity_kW * self.price_values[self.day - 1][
                self.hour - 1] * 2 * 1e-3
            
            # Reward shaping parameters
            valuation_battery = charged_electricity_kW*self.battery_valuation[int(self.state[5]-1)] * 1e-3
            
            penalty_morning = 0

            if self.hour == 7 and self.battery_level < 10:
                penalty_morning = 100
            if self.hour == 8 and self.battery_level < 20:
                penalty_8am = 100
                
            reward = -charged_electricity_costs + valuation_battery - penalty_morning
            
            self.battery_level += charged_electricity_kW * self.charge_efficiency

        # Calculate the profits and battery level when discharging (action <0)
        elif (action < 0) and (self.battery_level >= 0):
            if (self.battery_level + action * self.max_power) < 0:
                action = -self.battery_level / (self.max_power)
            discharged_electricity_kWh = action * self.max_power  # Negative discharge value
            discharged_electricity_profits = abs(discharged_electricity_kWh) * self.discharge_efficiency * \
                                             self.price_values[self.day - 1][self.hour - 1] * 1e-3
            # Reward shaping parameters
            valuation_battery = discharged_electricity_kWh*self.battery_valuation[int(self.state[5]-1)] * 1e-3
            
            
            reward = discharged_electricity_profits + valuation_battery
            
            self.battery_level += discharged_electricity_kWh
            # Some small numerical errors causing the battery level to be 1e-14 to 1e-17 under 0 :
            if self.battery_level < 0:
                self.battery_level = 0

        else:
            reward = 0

        self.counter += 1  # Increase the counter
        self.hour += 1  # Increase the hour

        if self.counter % 24 == 0:  # If the counter is a multiple of 24, increase the day, reset hour to first hour
            self.day += 1
            self.hour = 1

        terminated = self.counter == len(
            self.price_values.flatten()) - 1  # If the counter is equal to the number of hours in the train data, terminate the episode
        truncated = False

        info = action  # The final action taken after all constraints! For debugging purposes

        self.state = self.observation()  # Update the state

        return self.state, reward, terminated, truncated, info

    def observation(self):  # Returns the current state
        battery_level = self.battery_level
        price = self.price_values[self.day - 1][self.hour-1]
        hour = self.hour
        day_of_week = self.timestamps[self.day - 1].dayofweek  # Monday = 0, Sunday = 6
        day_of_year = self.timestamps[self.day - 1].dayofyear  # January 1st = 1, December 31st = 365
        month = self.timestamps[self.day - 1].month  # January = 1, December = 12
        year = self.timestamps[self.day - 1].year
        self.state = np.array(
            [battery_level, price, int(hour), int(day_of_week), int(day_of_year), int(month), int(year),
             int(self.car_is_available)])

        return self.state
    

    def reset(self):
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.car_is_available = True
        return self.state, 0