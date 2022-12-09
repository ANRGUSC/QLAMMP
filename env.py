from cProfile import label
import random
from turtle import color
from gym import Env
from gym.spaces import Discrete
import numpy as np
from curve import Curve
from user import User
from swap import Swap
from scipy.stats import truncnorm
from matplotlib import pyplot as plt 
import pandas as pd
from tqdm import tqdm


def getTruncatedNormal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class BothCustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(9)

        self.observation_space = Discrete(50000)

        self.state = random.randint(0, 49999)

        self.numUsers = 20
        self.numSwaps = self.numUsers * 20

        self.curve = None
        self.userNums = None
        self.users = None
        self.swaps = None

        self.bins = np.linspace(-100.0, 100.0, num=50000)
        
    # Execute one time step within the environment
    def step(self, action):        
        if action == 0:
            pass

        elif action == 1:
            if self.curve.getLev() < 84:
                self.curve.plusLev()

        elif action == 2:
            if self.curve.getLev() > 2:
                self.curve.minusLev()
        
        elif action == 3:
            if self.curve.getFee() < 30:
                self.curve.plusFee()
        
        elif action == 4:
            if self.curve.getFee() < 30:
                self.curve.plusFee()
            if self.curve.getLev() < 84:
                self.curve.plusLev()

        elif action == 5:
            if self.curve.getFee() < 30:
                self.curve.plusFee()
            if self.curve.getLev() > 2:
                self.curve.minusLev()

        elif action == 6:
            if self.curve.getFee() > 4:
                self.curve.minusFee()

        elif action == 7:
            if self.curve.getFee() > 4:
                self.curve.minusFee()
            if self.curve.getLev() < 84:
                self.curve.plusLev()

        elif action == 8:
            if self.curve.getFee() > 4:
                self.curve.minusFee()
            if self.curve.getLev() > 2:
                self.curve.minusLev()

        # Make one swap
        status = self.swapIt()

        # Calculate reward
        reward, currSlippage, numSwaps = self.curve.getFeeRewards()
        if status == 0 or status == -1:
            reward = status

        self.state = self.getDiscreteState(currSlippage)

        if numSwaps >= self.numSwaps: 
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    # Reset the state of the environment to an initial state
    def reset(self, loose):
        self.state = random.randint(0, 49999)
        self.setCurve(loose)
        return self.state

    def getDiscreteState(self, obs):
        if obs >= -100 and obs <= 100:
            return np.digitize(np.array([obs,]), self.bins) - 1
        else:
            print(f"Oh no! obs: {obs}")
            return random.randint(0, 49999)

    def setCurve(self, loose):
        numUsers = self.numUsers
        numSwaps = self.numSwaps

        if loose:
            X1 = getTruncatedNormal(mean = 0.75, sd = 0.75, low = 0.1, upp = 5)
        else:
            X1 = getTruncatedNormal(mean = 0.25, sd = 0.25, low = 0.1, upp = 5)        
        X2 = getTruncatedNormal(mean = np.log(1.5), sd = 0.25, low = np.log(1), upp = np.log(2))

        tolerance = X1.rvs(numSwaps)
        urgency = X2.rvs(numSwaps)

        self.userNums = []
        for i in range(0, numSwaps):
            self.userNums.append(random.randint(0, numUsers - 1))


        self.users = []
        for i in range(numUsers):
            user = User(3500, 3500, i)
            self.users.append(user)

        self.swaps = []
        for i in range(numSwaps):
            swap = Swap(tolerance[i], urgency[i], self.userNums[i])
            self.swaps.append(swap)

        fee = random.randint(4, 30)
        lev = random.randint(1, 85)

        self.curve = Curve([20000, 20000], lev, fee)        
        # self.curve = Curve([20000, 20000], 42, 17)


    def swapIt(self):
        if len(self.swaps):
            if self.swaps[0].new:
                status, amount, idx_in, tolerance, urgency = self.users[self.swaps[0].userNum].makeSwap(self.curve, self.swaps[0].tolerance, self.swaps[0].urgency, True)
            else:
                status, amount, idx_in, tolerance, urgency = self.users[self.swaps[0].userNum].makeSwap(self.curve, self.swaps[0].tolerance, self.swaps[0].urgency, False, self.swaps[0].amt, self.swaps[0].idx)
            
            self.swaps[0].tries += 1
            if self.swaps[0].tries >= 15:
                self.swaps.pop(0)
            else:
                if status == 0:
                    self.swaps[0].tolerance = tolerance
                    self.swaps[0].urgency = urgency
                    self.swaps[0].amt = amount
                    self.swaps[0].idx = idx_in
                    self.swaps[0].new = False

                    putSwap = self.swaps.pop(0)
                    self.swaps.insert(min(len(self.swaps) - 1, 9), putSwap)
                else:
                    self.swaps.pop(0)
            
            if status == 1:
                return amount
            elif status == 0:
                return 0
            else:
                return -1
        else:
            print("len swaps !> 0")


if __name__ == "__main__":
    env = BothCustomEnv()

    alpha = 0.02
    discount_factor = 0.99             
    epsilon = 1                  
    max_epsilon = 1
    min_epsilon = 0.001         
    decay = 0.01 

    train_episodes = 3000  
    test_episodes = train_episodes // 4  
    window_size = train_episodes // 20       
    max_steps = 200000

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    training_rewards = []
    actions_std = []  

    for episode in tqdm(range(train_episodes), desc ="Progress"):
        actions = []
        
        loose = False
        # if episode/train_episodes >= 0.5:
        #     loose = True
        state = env.reset(loose)    

        total_training_rewards = 0

        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1) 
            
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state,:])      
            else:
                action = env.action_space.sample()

            actions.append(action)
                
            new_state, reward, done, info = env.step(action)

            Q[state, action] = Q[state, action] + alpha * (reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action]) 

            total_training_rewards += reward      
            state = new_state         
            
            #Ending the episode
            if done == True:
                break
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
        
        training_rewards.append(total_training_rewards)
        actions_std.append(np.std(actions))


    av = sum(sorted(training_rewards, reverse=True)[:750])/750
    print ("Average Rewards: $" + str(av))

    plt.plot(actions_std, color='cornflowerblue', label='Standard Deviation')
    plt.ylabel('Actions Standard Deviation')
    plt.xlabel('Epoch')
    plt.ylim(0, 5)
    plt.grid(linestyle=':')
    plt.legend(loc='lower right')
    plt.show()

    # # av = np.mean(training_rewards)
    # # print ("Average Rewards: $" + str(av))

    # df = pd.DataFrame(training_rewards, columns = ['training_rewards'])

    # rewards = df['training_rewards']
    # average = rewards.rolling(window=window_size).mean()
    # plt.plot(training_rewards, 'k-', label='Rewards')
    # plt.plot(average, 'r-', label='Running Average')
    # plt.axhline(y=av, color='royalblue', linestyle='dotted', label='Third Quartile')
    # # plt.axhline(y=av, color='royalblue', linestyle='dotted', label='Overall Average')
    # plt.ylabel('Total Rewards')
    # plt.xlabel('Epoch')
    # plt.ylim(0, 8000)
    # plt.grid(linestyle=':')
    # plt.fill_between(average.index, 0, average, color='r', alpha=0.1)
    # plt.legend(loc='lower right')
    # plt.show()