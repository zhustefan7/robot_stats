import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix


data_path1 = "data/oakland_part3_am_rf.node_features"
data_path2 = "data/oakland_part3_an_rf.node_features"
label_order = np.array([1004,1100,1103,1200,1400])


def create_data_set(data_path1, data_path2):
    training_data = []
    testing_data = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_pos = []
    test_pos =[]
    data1 = np.loadtxt(data_path1)
    data2 = np.loadtxt(data_path2)
    data = np.vstack((data1,data2))
    np.random.shuffle(data)
    for i in range(data.shape[0]):
        is_train = np.random.choice([1,0], p=[0.8,0.2])
        if is_train:
            training_data.append(data[i,:])
            train_x.append(data[i,5:-2])
            train_y.append(data[i,4])
            train_pos.append(data[i,0:3])
        else:
            testing_data.append(data[i,:])
            test_x.append(data[i,5:-2])
            test_y.append(data[i,4])
            test_pos.append(data[i,0:3])
    training_data = np.array(training_data)
    testing_data = np.array(testing_data)
    np.savez("data/train.npz",x=train_x, y=train_y,pos= train_pos)
    np.savez("data/test.npz",x=test_x, y=test_y, pos= test_pos)
# return training_data, testing_data



#make prediction:
def multi_svm_predict(x_t, W_t):
    '''
    Multi-class svm prediction

    input: 
        x_t: d-dim feature vector
        W_t: the current weight matrix in dim (k X d):

    Output: 
        hat{y}_t: the prediction your svm algorithm would make
    '''

    #TODO: implement the prediction procedure:
    hat_y = np.argmax(np.matmul(W_t , x_t[:, np.newaxis]))
    # print(np.matmul(W_t , x_t[:, np.newaxis]).shape)
    return hat_y

#update Weight matrix:
def multi_svm_update(x_t, y_t, W_t,hat_y_t):
    '''
    Multi-class svm update procedure:
    
    input: 
            x_t: d-dim feature vector
            y_t: label, integer from [0,1,2..,k-2,k-1]
            W_t: the current Weight Matrix in dim (k X d)
    
    Output:
            W_tp1: the updated weighted matrix W_{t+1}.
    '''
   
    #TODO: implement the update rule to compute W_{t+1}:  
    rows, cols = W_t.shape

    Ut = np.zeros((rows,cols))
    # for i in range(rows):
    # if y_t*(np.matmul( W_t[y_t,:], x_t)) < 1:
    #     Ut[y_t,:] = y_t*x_t
        # Ut[hat_y_t] = -y_t*x_t
    for i in range(rows):
        if i==y_t:
            y= 1 
        else:
            y=-1
        if y*(np.matmul( W_t[i,:], x_t)) < 1:
            Ut[i,:] = x_t * y
    W_t = W_t + Ut
    return W_t


def train_online_svm(X, Y):
    '''
    We put every pieces we implemented above together in this function.
    This function simulates the online learning procedure.  
    (you do not need to implement anything here)

    input:
        X: N x d, where N is the number of examples, and d is the dimension of the feature.
        Y: N-dim vector.
            --Multi-Class: Y[i] is the label for example X[i]. Y[i] is an integer from [0,1.,..k-2,k-1] (k classes).
            --Binary: Y[i] is 0 or 1. 
    
    output: 
        M: a list: M[t] is the average number of mistakes we make up till and including time step t. Note t starts from zero.  
           you should expect M[t] decays as t increases....
           you should expect M[-1] to be around ~0.2 for the mnist dataset we provided. 

        W: final predictor.
            --Binary: W is a d-dim vector, 
            --Multi-Class: W is a k X d matrix. 
    '''

    d = X.shape[1]   #feature dimension.
    k = np.unique(Y).shape[0] #num of unique labels.  k=2: binary, k>2: multi-class
    M = []

    t_mistaks = 0
    #Initialization for W:
    W = np.zeros((k, d)) if k>2 else np.zeros(d) 
    epochs = 5
    #we scan example one by one:
    total_counter = 0
    for epoch in range(epochs):
        for t in range(X.shape[0]):
            total_counter +=1
            x_t = X[t]    #nature reveals x_t. 
            hat_y_t = multi_svm_predict(x_t, W) #we make prediction
            y_t = Y[t]   #nature reveals y_t after we make prediction.
            y_t = np.where(label_order==y_t)[0][0]
            # print(y_t)
            if y_t != hat_y_t: 
                t_mistaks = t_mistaks + 1
            
            W = multi_svm_update(x_t, y_t, W,hat_y_t) #svm update. 

            M.append(t_mistaks/float(total_counter) )  #record the average number of mistakes.
            # print(t_mistaks/float(total_counter))
    return M, W

def convert_label(Y):
    labels = []
    for i in range(Y.shape[0]):
        y_t = Y[i]
        labels.append(np.where(label_order==y_t)[0][0])
    return np.array(labels)



def validate(X,Y,W):

    d = X.shape[1]   #feature dimension.
    k = np.unique(Y).shape[0] #num of unique labels.  k=2: binary, k>2: multi-class
    M = []
    predicted_labels =[]
    t_mistaks = 0
    #we scan example one by one:
    total_counter = 0
    Y = convert_label(Y)
    for t in range(X.shape[0]):
        total_counter +=1
        x_t = X[t]    #nature reveals x_t. 
        hat_y_t = multi_svm_predict(x_t, W) #we make prediction
        y_t = Y[t]   #nature reveals y_t after we make prediction.
        # print(y_t)
        if y_t != hat_y_t: 
            t_mistaks = t_mistaks + 1
        predicted_labels.append(hat_y_t)
        M.append(t_mistaks/float(total_counter) )  #record the average number of mistakes.
        print(t_mistaks/float(total_counter))
    cm = confusion_matrix(Y, predicted_labels)
    return M,np.array(predicted_labels), cm




# def visualize(poses, labels):
#     fig = plt.figure()
#     # color = np.array([[0,255,0],[255,0,0],[0,0,255],[100,100,255],[255,100,100]])
    
#     color = ['r','b','y','g','m']
#     ax = fig.add_subplot(111, projection='3d')

#     color_array=[]
#     for i in range(poses.shape[0]):
#         label = np.where(label_order==labels[i])[0][0]
#         color_array.append(color[label])

#     ax.scatter(poses[:,0], poses[:,1], poses[:,2], c=color_array, s=0.5)
#     plt.show()


def visualize(poses, labels,cm):
    colors = ['green','pink','blue','yellow','red']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=130) 
    for i in range(5):
        color = colors[i]
        indices = np.where(labels==i)[0]
        pos = poses[indices]
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color, s=0.5)

    plt.show()

    plt.matshow(cm)
    plt.show()





if __name__=="__main__":
    # create_data_set(data_path1, data_path2)
    train_data = np.load("data/train.npz")
    test_data = np.load("data/test.npz")
    X = train_data["x"]
    Y = train_data["y"]
    # convert_label(Y)
    train_poses = train_data["pos"]

    test_X = test_data["x"]
    test_Y = test_data["y"]
    test_poses = test_data["pos"]


    M, W = train_online_svm(X, Y)
    _,test_labels, test_cm = validate(test_X, test_Y,W)
    # _,train_labels, train_cm= validate(X, Y, W)
    # visualize(train_poses, convert_label(Y), train_cm)
    visualize(test_poses,test_labels, test_cm)

    # visualize(train_poses, train_labels, train_cm)
