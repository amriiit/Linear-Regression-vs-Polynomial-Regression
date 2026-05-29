import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Features:
# [marketing_spend, team_size, years_active]

X_train = np.array([
    [10,5,1],
    [15,7,2],
    [20,8,2],
    [25,10,3],
    [30,12,4],
    [40,15,5],
    [50,18,6],
    [60,20,7],
    [75,25,8],
    [90,30,10]
])

# Profit

y_train = np.array([
    20,
    30,
    45,
    60,
    85,
    130,
    180,
    240,
    340,
    500
])

#function for normalization
def normalization(x_train):
    mean=np.mean(x_train,axis=0)
    sd=np.std(x_train,axis=0)
    x_norm=(x_train-mean)/sd
    return x_norm,mean,sd

#function to compute total cost 
def compute_cost(x_train,y_train,w,b):
    m=X_train.shape[0]
    total_cost=0
    for i in range(m):
        prediction=np.dot(x_train[i],w)+b
        error=-y_train[i]+prediction
        total_cost+=error**2
    total_cost=(total_cost/(2*m))  
    return total_cost

#function to compute total cost 
def compute_gradient(x_train,y_train,w,b):
    m,n=X_train.shape
    cost=0
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        prediction=np.dot(w,x_train[i])+b
        error=prediction-y_train[i]
        for j in range(n):
            dj_dw[j]+=error*x_train[i,j]
        dj_db+=error
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
#function to calculate gradient_descent
def gradient_descent(x_train,y_train,w,b,iterations,alpha):
    m,n=X_train.shape
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(x_train,y_train,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        cost=compute_cost(x_train,y_train,w,b)
        cost_history.append(cost)
    return w,b,cost_history

#initializing the parameters
w_init=np.zeros(X_train.shape[1])
b_init=0
alpha=0.01
x_train,mean,sd=normalization(X_train)
cost_his=[]
iterations=1000
w,b,cost_his=gradient_descent(x_train,y_train,w_init,b_init,iterations,alpha)

#plotting the curve for cost vs iterations
plt.figure()
plt.plot(cost_his)
plt.xlabel("Iterations")
plt.ylabel("Total Cost")
plt.title("Cost vs Iterations")
plt.grid(True)

#plotting spend vs profit
plt.figure()
plt.scatter(X_train[:,0],y_train)
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.title("Marketing spend vs Profit")
plt.grid(True)


#plotting size vs profit
plt.figure()
plt.scatter(X_train[:,1],y_train)
plt.xlabel("Size")
plt.ylabel("Profit")
plt.title("Team size vs Profit")
plt.grid(True)


#plotting years active vs profit
plt.figure()
plt.scatter(X_train[:,2],y_train)
plt.xlabel("Years Active")
plt.ylabel("Profit")
plt.title("Years Active vs Profit")
plt.grid(True)



#plotting marketing_spend vs team_size vs cost
fig = plt.figure()
ax = fig.add_subplot(   111,   projection='3d')
ax.scatter(X_train[:,0],X_train[:,1],y_train)
ax.set_xlabel("Marketing")
ax.set_ylabel("Team Size")
ax.set_zlabel("Profit")
ax.set_title("Marketing vs Team Size vs Profit")

#plotting team_size vs years active vs cost
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X_train[:,0],X_train[:,2],y_train)
ax.set_xlabel("Marketing Spend")
ax.set_ylabel("Years Active")
ax.set_zlabel("Profit")
ax.set_title("Marketing Spend vs Years Active vs Profit")

plt.show()