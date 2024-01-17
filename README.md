# PoRL
Project on Reinforcement Learning for VU Amsterdam course P3 '23-'24. 
Here we train an RL model to decide if it should sell the stored electricity or buy more

# Contents
- `data` contains training and validation data in .xlsx format
- `models` stores approaches we use:
  - [x] interface for approaches
  - [x] baseline approach (sell when you have something to sell and price is higher than threshold, buy when price is lower than other threshold)
  - [x] DQN
  - [ ] tabular method
- [ ] `results` will store the reports
- `EDA.ipynb` contains exploratory data analysis on training dataset and preprocessing of data (feature engineering, outlier removal)
- `preprocessing.ipynb` is a script to convert data from hours format to a single series of values
- [x] `main.ipynb` will store the main training script

# Contributions
- Andrey
  1. EDA
  2. DQN model
  3. ToDo
- Jutte
  1. Environment
  2. ToDo
- Kenji
  1. Report
  2. Environment
  3. ToDo
