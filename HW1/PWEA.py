import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def get_expert_advices(game_num, obs, with_obs):
    if not with_obs:
        expert1_advice = 1       #always predicts win
        expert2_advice = -1      #always predicts lose
        expert3_advice = -1 if game_num%2 == 1 else 1 #predicts win every other time
    else:
        expert1_advice = 1 if obs[0]==1 else -1  #predicts win if good weather
        expert2_advice = 1 if obs[1]==1 else -1  #predicts win if home game
        expert3_advice = 1 if (obs[0]==1 and obs[1]==1) else -1 #predicts win if sunny and home game
    
    expert_advices = np.array([expert1_advice,expert2_advice,expert3_advice])
    return expert_advices

####Randomly generates Observation
def get_obs():
    obs = np.random.choice(2, 3)  #[weather, game_loc, morale]
    return obs

def ground_truth(expert_advice, world, weight, obs):
    #Stochastic 
    if world ==0:
        label = np.random.choice([-1,1], size=1) 
    #deterministic
    elif world == 1:
        label = 1
    #adversarial
    elif world == 2:
        label = - np.sign(np.sum(np.multiply(weight, expert_advice)))
    
    #a world that favors weather
    elif world == 3:
        #if weather is good
        if obs[0]:
            label = (np.random.choice([-1,1], p=[0.1,0.9]))
        else:
            label = np.random.choice([-1,1], p=[0.9,0.1])
        
    #a world that favors game lo
    elif world == 4:
        #if game loc is good
        if obs[1]:
            label = np.random.choice([-1,1], p=[0.1,0.9])
        else:
            label = np.random.choice([-1,1], p=[0.9,0.1])
    
    #a world that favors game loc and weather
    elif world == 5:
            #if game loc is good
        if obs[1] and obs[0]:
            label = np.random.choice([-1,1], p=[0.1,0.9])
        else:
            label = np.random.choice([-1,1], p=[0.1,0.9])

    
    return label



##3.3 WMA 
def WMA(num_games, world, decay = 0.5, with_obs = False):
    weight = np.ones(3)
    avg_cum_regret = 0

    learner_cum_loss = 0
    expert_cum_loss = np.zeros(3)

    learner_cum_loss_arr = []
    expert_cum_loss_arr = []
    avg_cum_regret_arr = []
    
    for i in range(num_games):
        #receive observation. the observation is only used if with_obs flag is true
        obs = get_obs()
        #receive expert advice
        expert_advice = get_expert_advices(i, obs, with_obs)
        #learner makes prediction based on expert advice and weight
        prediction = np.sign(np.sum(np.multiply(weight, expert_advice)))
        #received ground truth label
        label  = ground_truth(expert_advice, world, weight, obs)

        #update weight
        indicator = expert_advice!=label
        indicator = indicator.astype(int)
        weight = np.multiply(weight, 1- decay*indicator)

        #calculate loss and regret
        learner_loss = int(prediction != label)
        expert_loss = 1*indicator
        learner_cum_loss += learner_loss
        expert_cum_loss += expert_loss 
        avg_cum_regret = 1/float((i+1)) * (learner_cum_loss - np.min(expert_cum_loss))

        learner_cum_loss_arr.append(learner_cum_loss)
        expert_cum_loss_arr.append(expert_cum_loss.copy())
        avg_cum_regret_arr.append(avg_cum_regret)


    return np.array(learner_cum_loss_arr), np.array(expert_cum_loss_arr), np.array(avg_cum_regret_arr)


##3.4 RWMA 
def RWMA(num_games, world, decay = 0.5 ,with_obs = False):
    weight = np.ones(3)
    avg_cum_regret = 0

    learner_cum_loss = 0
    expert_cum_loss = np.zeros(3)

    learner_cum_loss_arr = []
    expert_cum_loss_arr = []
    avg_cum_regret_arr = []

    for i in range(num_games):
        #receive observation. the observation is only used if with_obs flag is true
        obs = get_obs()
        #receive expert advice
        expert_advice = get_expert_advices(i, obs, with_obs)
        
        #normalize weight 
        phi = np.sum(weight)
        probablity = weight/phi
        #make prediction using multinomial sampling based on the calculated probability
        index = np.where(np.random.multinomial(1, probablity) ==1)[0][0]
        prediction = expert_advice[index]

        #receives ground truth label
        label  = ground_truth(expert_advice, world, weight, obs)
        indicator = expert_advice!=label
        indicator = indicator.astype(int)
        #update weight
        weight = np.multiply(weight, 1- decay*indicator)

        learner_loss = int(prediction != label)
        expert_loss = 1*indicator
        learner_cum_loss += learner_loss
        expert_cum_loss += expert_loss 

        avg_cum_regret = 1/float((i+1)) * (learner_cum_loss - np.min(expert_cum_loss))

        learner_cum_loss_arr.append(learner_cum_loss)
        expert_cum_loss_arr.append(expert_cum_loss.copy())
        avg_cum_regret_arr.append(avg_cum_regret)


    return np.array(learner_cum_loss_arr), np.array(expert_cum_loss_arr), np.array(avg_cum_regret_arr)

def plot():
    num_games = 1000
    mode = "RWMA"
    with_obs = False
    decay = 0.01

    for world in range(3):
        if mode == "WMA":
            print('here')
            learner_cum_loss_arr, expert_cum_loss_arr, avg_cum_regret_arr = WMA(num_games, world, decay, with_obs)

        elif mode == "RWMA":
            learner_cum_loss_arr, expert_cum_loss_arr, avg_cum_regret_arr = RWMA(num_games, world, decay, with_obs)  
            
        x_axis = np.linspace(1, num_games, num=num_games)
        plt.figure(world,figsize=(8.0, 6.0))
        plt.plot(x_axis, avg_cum_regret_arr)
        plt.xlabel("T")
        plt.ylabel("Regret")
        plt.title("Average Cummulative Regret")
        # plt.savefig('/home/stefanzhu/Documents/2020_Fall/16831_robo_stats/HW1/imgs/world_{}_{}_regret.png'.format(world,mode),format='png')


        plt.figure(world+1,figsize=(8.0, 8.0)) 
        plt.subplot(211)
        plt.plot(x_axis , learner_cum_loss_arr)
        plt.title('Learner Cummulative Loss')

        plt.subplot(212)
        # print(expert_cum_loss_arr.shape)
        plt.plot(x_axis , expert_cum_loss_arr[:,0])
        plt.plot(x_axis , expert_cum_loss_arr[:,1])
        plt.plot(x_axis , expert_cum_loss_arr[:,2])
        plt.legend(['Exp1', 'Exp2', 'Exp3'])
        plt.xlabel("T")
        plt.ylabel("Loss")
        plt.title('Expert Cummulative Loss')
        # plt.savefig('/home/stefanzhu/Documents/2020_Fall/16831_robo_stats/HW1/imgs/world_{}_{}_loss.png'.format(world,mode),format='png')

        plt.show()


if __name__ == "__main__":
    plot()




