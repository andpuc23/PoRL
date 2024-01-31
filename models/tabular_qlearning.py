import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from envs.train_env import Electric_Car 
from envs.TestEnv import Electric_Car as Electric_Car_Test
import torch
import matplotlib.patches as mpatches

class TabularQLearning():
    def __init__(self, data_path_train, discount_rate, bin_size, state_vars_qtable):
        
        '''
        Params:
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        self.discount_rate = discount_rate
        self.bin_size = bin_size
        self.env = Electric_Car(path_to_train_data=data_path_train)
        self.state_vars_qtable = state_vars_qtable
        
        # Discretize the action space
        self.action_space = np.linspace(-1, 1, self.bin_size[-1]-1)
        self.bins = []
 
        for i in range(len(self.state_vars_qtable)):
            if self.state_vars_qtable[i] == 0: # Battery 
                bins_feature = np.linspace(0, self.env.battery_capacity, self.bin_size[i])
                bins_feature[-1]+=0.1
            elif self.state_vars_qtable[i]==1: # Price 
                bins_feature = np.linspace(0, np.percentile(self.env.price_values, 90), self.bin_size[i]-1)
            elif self.state_vars_qtable[i]==2: # Hour linear between 1 and 24
                bins_feature = np.linspace(1, 24, self.bin_size[i])
                bins_feature[-1]+=0.1
            elif self.state_vars_qtable[i]==3: #Days of the week, weekday or weekend
                bins_feature = np.array([0, 5, 7])
            elif self.state_vars_qtable[i]==5: # Month linear between 1 and 12
                bins_feature = np.linspace(1, 12, self.bin_size[i])
                bins_feature[-1]+=0.1
            self.bins.append(bins_feature) 
        
    def discretize_state(self, state):
        
        '''
        Params:
        state = state observation that needs to be discretized
        
        
        Returns:
        discretized state
        '''
        
        self.state = state
        digitized_state = []
        
        # Change hour (index 2) from [1, ..., 24] to [0, ..., 23]
        # Change month (index 5) from [1, ..., 12] to [0, ..., 11]
        #digitized_state[2]-=1
        #digitized_state[5]-=1
        
        # Discretize the features of the state
        bins_idx = 0
        for i in range(len(self.state)):
            if i in self.state_vars_qtable:
                digitized_state.append(np.digitize(self.state[i], self.bins[bins_idx])-1)
                bins_idx+=1
            else:
                digitized_state.append(int(self.state[i]))
        
        #for i in range(len(self.bins)):
        #    digitized_state.append(np.digitize(self.state[i], self.bins[i])-1)

        # Add the other variables
        #digitized_state.extend(state[len(self.bins):])
        #digitized_state = np.array(digitized_state).astype(int)
        
        return np.array(digitized_state)
    
    def create_Q_table(self):
        '''
        Returns:
        Q-table with zeros
        '''
        self.state_space = np.array(self.bin_size)-1
        self.Qtable = np.zeros(self.state_space)
        self.Qtable_updates = self.Qtable.copy()

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
                self.Qtable_updates[tuple(state[self.state_vars_qtable]) + (idx_action,)] += 1
                
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
        return self.Qtable, self.Qtable_updates, self.rewards, self.average_rewards
        
    def visualize_rewards_and_performance(self):
        zero = np.count_nonzero(self.Qtable ==0)
        percent_zero = zero / self.Qtable.size * 100
        print("The Q table has", zero, "zero values which accounts to", percent_zero, "%")

        plt.figure(figsize =(7.5,7.5))
        plt.plot(self.sims_per_avg*(np.arange(len(self.average_rewards))+1), self.average_rewards)
        #plt.axhline(y = -110, color = 'r', linestyle = '-')
        plt.title('Average reward over the past" %d simulations'%self.sims_per_avg, fontsize = 10)
        #plt.legend(['Q-learning performance','Benchmark'])
        plt.xlabel('Number of simulations', fontsize = 10)
        plt.ylabel('Average reward', fontsize = 10)
        plt.show()
        
    def play(self, data_test, plotting = True):
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
            states.append(state)
            state = self.discretize_state(state)
            idx_action = np.argmax(self.Qtable[tuple(state[self.state_vars_qtable])])
            action = self.action_space[idx_action]
            next_state, reward, terminated, truncated, info = test_env.step(action)
            #next_state = self.discretize_state(next_state)
            done = terminated or truncated
            state = next_state
            actions.append(action)
            infos.append(info)
            rewards.append(reward)

        #test_env.close()   
        #plt.scatter(range(len(rewards)), rewards)
        #plt.title('Rewards durnig test')
        #plt.show()
        if (plotting ==True):
            self.plot(actions, rewards, states, infos)

        return actions, rewards, states, infos
    

    def save_Qtable(self, file):
        np.save(file, self.Qtable)
        return 
    
    def load_Qtable(self, file):
        self.Qtable = np.load(file)
        return 
    

    def plot(self, actions, rewards, states, infos, max = 400):
        batteries = np.array(states)[:, 0]
        prices = np.array(states)[:, 1]
        x = range(len(prices))
        hours = np.array(states)[:, 2]


        #The following code creates boolean arrays indicating what the final action is we take, forced buying happens when we do not have 20KwH at 7AM
        #Needed for colour coding scatter plots conditionally
        buying = np.full(len(batteries), False, dtype=bool)
        selling = np.full(len(batteries), False, dtype=bool)
        forced_charging = np.full(len(batteries), False, dtype=bool)
        for i in range(len(batteries - 1)):
            if actions[i] > 0 and infos[i] > 0:
                buying[i] = True
            elif infos[i] > 0:
                forced_charging[i] = True
            elif infos[i] < 0:
                selling[i] = True


        #Plotting the prices with the actions
        plt.figure(figsize=(12, 6))
        plt.ylim(0, 300)
        plt.xlim(0, max)
        col = np.where(forced_charging,'darkred', np.where(buying, 'red', np.where(selling, 'green', '#FF000000')))
        plt.scatter(x, prices, c = col,s = 30, zorder=1)
        plt.plot(x, prices, zorder =0, alpha=0.8)
        plt.ylabel('Price in Euro')
        plt.xlabel('Time')
        custom_legend = [
                mpatches.Patch(color='green', label='Selling'),
                mpatches.Patch(color='red', label='Buying'),
                mpatches.Patch(color='darkred', label='Forced Charging')
                ]
        plt.legend(handles=custom_legend)
        plt.title('Price and Policy')
        plt.show()


        #The following code creates arrays between timestamps indicating the action taken
        #Needed to make linecharts with conditional colours
        f = np.array([False])
        prev_fc = np.concatenate((f, forced_charging[:-1]), dtype = bool)
        prev_buy = np.concatenate((f, buying[:-1]), dtype = bool)
        prev_sell = np.concatenate((f, selling[:-1]), dtype = bool)
        prev_fc = np.logical_or(forced_charging, prev_fc)
        prev_buy = np.logical_or(buying, prev_buy)
        prev_sell = np.logical_or(selling, prev_sell)
        fc = batteries.copy()
        buy = batteries.copy()
        sell = batteries.copy()
        for i in range (len(batteries)):
            if prev_fc[i] == False:
                fc[i] = None
            if prev_buy[i] == False:
                buy[i] = None
            if prev_sell[i] == False:
                sell[i] = None


        #Plotting the Battery and actions
        plt.figure(figsize=(12, 6))
        plt.ylim(0, 55)
        plt.xlim(0, max)
        col = np.where(forced_charging,'darkred', np.where(buying, 'red', np.where(selling, 'green', '#FF000000')))
        plt.scatter(x, batteries, c = col, s = 30, zorder=1)
        plt.plot(x, batteries, zorder = 0)
        plt.plot(x, buy, alpha=0.8, c = 'red', zorder =1)
        plt.plot(x, sell, alpha=0.8, c = 'green', zorder =1)
        plt.plot(x, fc, alpha = 0.8, c = 'darkred', zorder = 1, label='Battery and actions')
        plt.ylabel('Battery')
        plt.xlabel('Time')
        plt.legend(handles=custom_legend)
        plt.title('Battery and Policy')
        plt.show()

        plt.figure(figsize=(12,6))
        hours_forced = np.zeros(24)
        hours_buying = np.zeros(24)
        hours_selling = np.zeros(24)
        for i in range(len(batteries)):
            if forced_charging[i] == True:
                hours_forced[int(hours[i])-1] += 1
            if buying[i] == True:
                hours_buying[int(hours[i])-1] += 1
            if selling[i] == True:
                hours_selling[int(hours[i])-1] += 1

        barWidth = 0.3
        xpos = np.arange(len(hours_forced))
        plt.bar(xpos+1.3, hours_forced, width = barWidth, color="darkred", label = 'Forced Charging')
        plt.bar(xpos+1, hours_buying, width = barWidth,  color="red", label = 'Buying')
        plt.bar(xpos+0.7, hours_selling, width = barWidth,  color="green", label = 'selling')
        plt.xticks(xpos+1)
        plt.legend()
        plt.xlabel('Hour')
        plt.ylabel('Count of Actions')
        plt.title('Actions per Hour')
        plt.show()

        #Plotting the cumlative reward in given timeframe
        plt.figure(figsize=(12, 6))
        cum_rewards = np.cumsum(np.array(rewards))
        plt.xlim(0, max)
        plt.ylim(-100, 10)
        plt.plot(x, cum_rewards, label = 'Cumulative Rewards')
        plt.xlabel('Time')
        plt.ylabel('Profit in Euro')
        plt.title('Cumulative rewards up to t =%i' %max)
        plt.show()


        #Plotting the Total Cumulative Reward
        plt.figure(figsize=(12, 6))
        cum_rewards = np.cumsum(np.array(rewards))
        plt.plot(x, cum_rewards)
        plt.title('Total Cumulative Rewards | Value = %.2f' % sum(rewards))
        plt.xlabel('Time')
        plt.ylabel('Profit in Euro')
        plt.show()

