from models.interface import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class BaselineModel(Model):
    def __init__(self, env):      
        self.lower_thres = np.quantile(env.price_values, 0.2)
        self.upper_thres = np.quantile(env.price_values, 0.6)
        self.env = env
        return
    
    def set_thres(self, lower, upper):
        self.lower_thres = np.quantile(self.env.price_values, lower)
        self.upper_thres = np.quantile(self.env.price_values, upper) 
        return    
        
    def predict(self, state):
        last_price = state[1]

        if self.lower_thres > last_price:
            action = 1 # buy max 25 cheap
        elif self.upper_thres < last_price:
            action = -1 
        else:
            action = 0 # do nothing

        return action

    def learn(self):
        states, rewards, infos, actions = [], [], [], []
        truncated = False
        terminated = False
        self.env.reset()
        state = self.env.observation()
        while(terminated == False and truncated == False):
            action = self.predict(state)
            obs, reward, termination, truncation, info = self.env.step(action)

            states.append(state)
            rewards.append(reward)
            terminated = termination
            truncated = truncation
            actions.append(action)
            infos.append(info)
            state = obs


        return states, rewards, terminated, truncated, infos, actions

    def train(self):
        lower_thres = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        upper_thres = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        #lower_thres = [0.4]
        #upper_thres = [0.8]
        results = [] 
        for i in range(len(lower_thres)):
            for j in range(len(upper_thres)):
                self.set_thres(lower_thres[i], upper_thres[j])
                states, rewards, terminated, truncated, infos, actions = self.learn()
                results.append([sum(rewards), lower_thres[i], upper_thres[j]])
                
        max_idx = np.argmax(np.array(results)[:,0])
        self.set_thres(results[max_idx][1], results[max_idx][2])
        print('The best thresholds are:', self.lower_thres, ' and ', self.upper_thres, 'with the following total reward: ', results[max_idx][0])
        print('these are now the selected thresholds')
        return

    def play_game_and_plot (self, max):

        states, rewards, terminated, truncated, infos, actions = self.learn()
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
        upper_line = np.full(shape=len(states), fill_value=self.upper_thres, dtype=np.float64)
        lower_line = np.full(shape=len(states), fill_value=self.lower_thres, dtype=np.float64)
        plt.plot(x, upper_line, c= "green", alpha=0.5)
        plt.plot(x, lower_line, c= "red", alpha=0.5, zorder=1)
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






