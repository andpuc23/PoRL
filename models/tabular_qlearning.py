import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from envs.train_env import Electric_Car 
from envs.test_env import Electric_Car as Electric_Car_Test
import torch

class TabularQLearning():
    def __init__(self, data_path_train, discount_rate, bin_size):
        
        '''
        Params:
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        self.discount_rate = discount_rate
        self.bin_size = bin_size
        self.env = Electric_Car(path_to_test_data=data_path_train)
        
        # Discretize the action space into [-1.0, -0.9, ..., 1]
        self.action_space = np.linspace(self.env.continuous_action_space.low[0], self.env.continuous_action_space.high[0], 21)
 
        # Bins of battery are linear from 0 until 50 kWh
        self.bins_battery = np.linspace(0, self.env.battery_capacity, self.bin_size) 
        # Bins of price are linear between 0 and the 0.9 quantile of all prices, the last bin contain all higher values
        self.bins_price = np.append(np.linspace(0, np.percentile(self.env.price_values, 90), self.bin_size-1),np.max(self.env.price_values)) 
        self.bins = [self.bins_battery, self.bins_price]
        
    
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
        
        # Add the other variables
        digitized_state.extend(state[len(self.bins):])
        digitized_state = np.array(digitized_state).astype(int)
        
        # Change hour (index 2) from [1, ..., 24] to [0, ..., 23]
        # Change month (index 5) from [1, ..., 12] to [0, ..., 11]
        digitized_state[2]-=1
        digitized_state[5]-=1
        
        return digitized_state
    
    def create_Q_table(self):
        '''
        Returns:
        Q-table with zeros
        '''
        
        self.state_space = self.bin_size - 1
        self.state_vars_qtable = [0, 1, 2, 3, 5] # Indices of variables used in the Q-table
       
        self.Qtable = np.zeros((self.bin_size, self.bin_size, 24, 7, 12, 21))

    def train(self, simulations, simulations_per_avg, learning_rate, epsilon, epsilon_decay, adaptive_epsilon, 
              adapting_learning_rate):
        
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
        self.epsilon_end = epsilon
        self.sims_per_avg = simulations_per_avg

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

            if i % self.sims_per_avg == 0:
                self.average_rewards.append(np.mean(self.rewards))
                self.rewards = []
            
        print('The simulation is done!')
        return self.Qtable, self.rewards, self.average_rewards
        
    def visualize_rewards(self):
        plt.figure(figsize =(7.5,7.5))
        plt.plot(self.sims_per_avg*(np.arange(len(self.average_rewards))+1), self.average_rewards)
        #plt.axhline(y = -110, color = 'r', linestyle = '-')
        plt.title('Average reward over the past 100 simulations', fontsize = 10)
        #plt.legend(['Q-learning performance','Benchmark'])
        plt.xlabel('Number of simulations', fontsize = 10)
        plt.ylabel('Average reward', fontsize = 10)
        plt.show()
            
    def play(self, data_test):
        # Make eval env which renders when taking a step
        test_env = Electric_Car_Test(data_test)
        state = test_env.reset()[0]
        states = []
        rewards = []
        actions = []
        infos = []
        done=False
        
        # Run the environment for 1 episode
        while not done:
            state = test_env.observation()
            state.append(state)
            state = self.discretize_state(state)
            idx_action = np.argmax(self.Qtable[tuple(state[self.state_vars_qtable])])
            action = self.action_space[idx_action]
            next_state, reward, terminated, truncated, info = test_env.step(action)
            #next_state = self.discretize_state(next_state)
            done = terminated or truncated
            state = next_state
            actions.append(action)
            infos.append(infos)
            rewards.append(reward)
            states
        #test_env.close()   
        #plt.scatter(range(len(rewards)), rewards)
        #plt.title('Rewards durnig test')
        #plt.show()
        return actions, rewards, states, infos
    
    def save_Qtable(self, file):
        np.save(file, self.Qtable)
        return 
    
    def load_Qtable(self, file):
        self.Qtable = np.load(file)
        return 
