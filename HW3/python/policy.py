import numpy as np

class Policy:
    """
    DO NOT MODIFY
    """
    def init(self, nbActions):
        self.nbActions = nbActions
    def decision(self):
        pass
    def getReward(self,reward):
        pass

class policyRandom(Policy):
    """
    DO NOT MODIFY
    """
    def decision(self):
        return np.random.randint(0,self.nbActions,dtype=np.int)
    def getReward(self,reward):
        pass

class policyConstant(Policy):
    """
    DO NOT MODIFY
    """
    def init(self,nbActions):
        self.chosenAction = np.random.randint(0,nbActions,dtype=np.int)
    def decision(self):
        return self.chosenAction
    def getReward(self,reward):
        pass

class policyGWM(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.reward = 0 
        self.weight = np.ones(nbActions)
        self.loss = np.zeros(nbActions)
        self.prob = np.zeros(nbActions)
        self.action = None
        self.T = 0

    def decision(self):
        self.T +=1
        self.prob  = self.weight/np.sum(self.weight)
        self.action = np.where(np.random.multinomial(1, self.prob ) ==1)[0][0]
        return self.action

    def getReward(self,reward):
        self.loss = np.zeros(self.nbActions)
        self.reward = reward
        self.loss[self.action] = 1- self.reward
        eta = np.sqrt(np.log(self.nbActions)/self.T)
        self.weight = self.weight * np.exp(-eta*self.loss)

        

class policyEXP3(Policy):
    """
    DO NOT MODIFY
    """
    def init(self, nbActions):
        self.nbActions = nbActions
        self.reward = 0 
        self.weight = np.ones(nbActions)
        self.loss = np.zeros(nbActions)
        self.prob = np.zeros(nbActions)
        self.action = None
        self.T = 0
    def decision(self):
        self.T +=1
        self.prob = self.weight/np.sum(self.weight)
        self.action = np.where(np.random.multinomial(1, self.prob) ==1)[0][0]
        return self.action
    def getReward(self,reward):
        self.loss = np.zeros(self.nbActions)
        self.reward = reward
        self.loss[self.action] = 1- self.reward
        self.loss = self.loss/self.prob
        eta = np.sqrt(np.log(self.nbActions)/(self.T*self.nbActions))
        self.weight = self.weight * np.exp(-eta*self.loss)
        

class policyUCB(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.reward = 0 
        self.action_counter = np.zeros(nbActions)
        self.cum_rewards = np.zeros(nbActions)
        self.loss = np.zeros(nbActions)
        self.action = None
        self.T = 1
        self.alpha = 1


    def decision(self):
        if self.T <= self.nbActions:
            self.action = self.T -1 
        else:
            self.action = np.argmax(self.cum_rewards/(self.T) + np.sqrt(self.alpha * np.log(self.T)/(2*self.action_counter))) 
        self.action_counter[self.action] +=1
        self.T +=1
        return self.action

    def getReward(self,reward):
        self.cum_rewards[self.action]+=reward