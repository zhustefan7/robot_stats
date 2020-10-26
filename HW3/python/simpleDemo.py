import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from policy import *
from game import *

game = gameConstant()
# game = gameGaussian(10,1000)
game = gameLookupTable('../data/univLatencies.mat', True)

# policies = [policyRandom(),policyConstant()]
# policy_names = ['policyRandom','policyConstant']

# policies = [policyGWM(), policyEXP3()]
# policy_names = ["policyGWM", 'policyEXP3']

policies = [policyUCB()]
policy_names = ["policyUCB"]

fig1 = plt.figure()

for k in range(len(policies)):
    game.resetGame()
    reward,action,regret = game.play(policies[k])
    print("{} Reward {:.2f}".format(policy_names[k],reward.sum()))
    plt.plot(regret,label=policy_names[k])
    plt.xlabel('trials')
    plt.ylabel('regret')
plt.legend()

fig2 = plt.figure()
for k in range(len(policies)):
    game.resetGame()
    reward,action,regret = game.play(policies[k])
    print("{} Reward {:.2f}".format(policy_names[k],reward.sum()))
    plt.plot(action,label=policy_names[k])
    plt.xlabel('trials')
    plt.ylabel('action')
plt.legend()

plt.show()
