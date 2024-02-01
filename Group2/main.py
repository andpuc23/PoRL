from TestEnv import Electric_Car
import argparse
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent


# Make the excel file as a command line argument, so that you can do: " python3 main.py --excel_file validate.xlsx "
parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = Electric_Car(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []
battery_level = []

RL_agent=Agent()
RL_agent.train('train.xlsx')


observation = env.observation()
for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    action = RL_agent.act(observation)
    
    # The observation is the tuple: [battery_level, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))
    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.ylabel('Cumulative reward (€)')
        plt.show()



