import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from envs.train_env import Electric_Car

class TabularQLearning():
    def __init__(self, data_path, discount_rate = 0.2, bin_size = 20):
        
        '''
        Params:
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        self.discount_rate = discount_rate
        self.bin_size = bin_size
        self.env = Electric_Car(path_to_test_data=data_path)
        self.action_space = np.linspace(self.env.continuous_action_space.low[0], self.env.continuous_action_space.high[0], 21)
        #self.action_space = self.env.continuous_action_space
        #self.bins_action_space = np.linspace(self.env.continuous_action_space.low[0], self.env.continuous_action_space.high[0], self.bin_size)
        
        self.bins_battery = np.linspace(0, self.env.battery_capacity, self.bin_size) 
        self.bins_price = np.append(np.linspace(0, np.percentile(self.env.price_values, 90), self.bin_size-1),np.max(self.env.price_values)) 
        self.bins = [self.bins_battery, self.bins_price]
        
        #print('bins_actions', self.bins_action_space)
        #print('bins_battery', self.bins_battery)
        #print('bins_price', self.bins_price)
        #print('bins', self.bins)
    
    def discretize_state(self, state):
        
        '''
        Params:
        state = state observation that needs to be discretized
        
        
        Returns:
        discretized state
        '''
        
        self.state = state
        digitized_state = []
        
        # Discretize the continuous variables battery (index 0) and price (index 1)
        for i in range(len(self.bins)):
            digitized_state.append(np.digitize(self.state[i], self.bins[i])-1)
        
        digitized_state.extend(state[len(self.bins):])
        digitized_state = np.array(digitized_state).astype(int)
        
        # Change hour (index 2): 1, ..., 24 to 0, ..., 23
        # Change month (index 5): 1, ..., 12 to 0, ..., 11
        digitized_state[2]-=1
        digitized_state[5]-=1
        
        return digitized_state
    
    def discretize_action(self, action): 
        '''
        Params:
        action = action that needs to be discretized
        
        Returns:
        discretized action
        '''
        
        self.action = action 
        digitized_action = np.digitize(self.action, self.bins_action_space)-1
        
        return digitized_action
    
    def create_Q_table(self):
        '''
        Returns:
        Q-table with zeros
        '''
        
        self.state_space = self.bin_size - 1
        self.state_vars_qtable = [0, 1, 2, 3, 5] # Indices of variables used in the Q-table
        
        #self.Qtable = np.zeros((self.state_space, self.state_space, 24, 7, 365, 12, 3, 2, self.state_space)) 
        #self.Qtable = np.zeros((self.bin_size, self.bin_size, 24, 7, 12, self.bin_size)) 
        self.Qtable = np.zeros((self.bin_size, self.bin_size, 24, 7, 12, 21))

    def train(self, simulations, learning_rate, epsilon = 0.05, epsilon_decay = 100, adaptive_epsilon = True, 
              adapting_learning_rate = False):
        
        '''
        Params:
        
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        
        '''
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.epsilon_start = 1
        self.epsilon_end = 0.05

        self.rewards = []
        self.average_rewards = []
        self.create_Q_table()
        
        if adapting_learning_rate:
            self.learning_rate = 1
        
        for i in range(simulations):
            #if i % 5000 == 0:
            #    print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            
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
                    idx_action = np.argmax(self.Qtable[tuple(state[self.state_vars_qtable])])
                    action = self.action_space[idx_action]
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                next_state = self.discretize_state(next_state)
                done =  terminated or truncated
                idx_action = np.where(self.action_space==action)
               
                Q_target = (reward + self.discount_rate*np.max(self.Qtable[tuple(next_state[self.state_vars_qtable])]))
                
                delta = self.learning_rate * (Q_target - self.Qtable[tuple(state[self.state_vars_qtable]) + (idx_action,)])
                
                self.Qtable[tuple(state[self.state_vars_qtable]) + (idx_action,)] = self.Qtable[tuple(state[self.state_vars_qtable]) + (idx_action,)] + delta
                
                total_rewards += reward
                state = next_state
            
            if adapting_learning_rate:
                self.learning_rate = self.learning_rate/np.sqrt(i+1)
            
            self.rewards.append(total_rewards)
            print(total_rewards)
            print(self.epsilon)

            if i % 100 == 0:
                self.average_rewards.append(np.mean(self.rewards))
                self.rewards = []
            
        print('The simulation is done!')
        return self.Qtable
        
    def visualize_rewards(self):
        pass
            
    def play_game(self):
        pass
