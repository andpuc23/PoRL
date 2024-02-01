import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
#from train_env import Electric_Car 
import torch
import matplotlib.patches as mpatches
import pandas as pd
from utils import Electric_Car_Train

class Agent():
    def __init__(self):
        
        '''
        Params:
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        self.discount_rate = 0.975
        self.bin_size = [4, 6, 7, 3, 4, 4]  
        self.features_qtable = [0, 1, 2, 3, 5]
        self.make_bins()
        self.action_space = np.linspace(-1, 1, self.bin_size[-1]-1)
        
        # Discretize the action space and the state space of specified features

        
        
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
    
    
    def make_bins(self):
        self.bins = []
 
        for i in range(len(self.features_qtable)):
        
            if self.features_qtable[i] == 0: # Battery 
                bins_feature = np.linspace(0, 50, self.bin_size[i])
                bins_feature[-1]+=0.1
                
            elif self.features_qtable[i]==1: # Price 
                bins_feature = np.linspace(0, 88, self.bin_size[i]-1)
            
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

    def train(self, data_path_train):
        
        '''
        Params:
        
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        
        '''
        self.env = Electric_Car_Train(path_to_train_data=data_path_train)
        self.action_space = np.linspace(-1, 1, self.bin_size[-1]-1)
        self.make_bins()        


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
                #next_state = .process_data(next_state)
                
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
        obs = self.discretize_state(obs)
            
        # Find optimal action
        idx_action = np.argmax(self.Qtable[tuple(obs[self.features_qtable])])
        action = self.action_space[idx_action]

        return action

    def save_Qtable(self, file):
        np.save(file, self.Qtable)
        return 
    
    def load_Qtable(self, file):
        self.Qtable = np.load(file)
        return 
