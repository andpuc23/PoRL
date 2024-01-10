import gym
from gym import Discrete


PUNISHENT = -100


class Environment(gym.Env):
    def __init__(self, data) -> None:
        super().__init__()
        self.action_space = Discrete(3)
        
        self.TOTAL_VOLUME = 50 # max charge
        self.data = data
        self.reset()


    def reset(self):
        self.charge = 20
        self.is_available = True
        
        


    def step(self, action):
        """
        action: float -25 to 25, negative is to sell, positive is to buy
        """
        pass            


