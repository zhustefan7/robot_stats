import numpy as np
from scipy.io import loadmat

class Game:
    nbActions = 0
    totalRounds = 0
    N = 0
    tabR = np.array([])

    def __init__(self):
        return
    def play(self,policy):
        policy.init(self.nbActions)
        reward = np.zeros(self.totalRounds)
        action = np.zeros(self.totalRounds,dtype=np.int)
        regret = np.zeros(self.totalRounds)

        for t in range(self.totalRounds):
            action[t] = policy.decision()
            reward[t] = self.reward(action[t])
            regret[t] = self.cumulativeRewardBestActionHindsight(t) - sum(reward)
            policy.getReward(reward[t])
            self.N += 1
        return reward, action, regret
    def reward(self,a):
        return self.tabR[a, self.N]
    def resetGame(self):
        self.N = 0
    def cumulativeRewardBestActionHindsight(self,t):
        # TODO
        cum_best_reward = 0
        for i in range(t):
            cum_best_reward += np.max(self.tabR[:,i])
        return cum_best_reward

class gameConstant(Game):
    """
    DO NOT MODIFY
    """
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.tabR = np.ones((2,1000))
        self.tabR[0] *= 0.8
        self.tabR[1] *= 0.2
        self.N = 0

class gameGaussian(Game):
    def __init__(self,nbActions,totalRound):
        super().__init__()
        self.nbActions = nbActionsq
        self.totalRounds = totalRound
        self.N = 0
        self.mean = np.random.random_sample()
        self.std = np.random.random_sample()
        self.tabR = np.random.normal(self.mean, self.std, (nbActions,totalRound))
        self.tabR = (self.tabR- np.min(self.tabR))/( np.max(self.tabR) - np.min(self.tabR))


class gameAdverserial(Game):
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.N = 0

class gameLookupTable(Game):
    def __init__(self,tabInput,isLoss):
        super().__init__()
        self.N = 0
        self.tabR = loadmat(tabInput)['univ_latencies']
        if isLoss:
            self.tabR = 1-self.tabR
            self.totalRounds = self.tabR.shape[1]
            self.nbActions = self.tabR.shape[0]