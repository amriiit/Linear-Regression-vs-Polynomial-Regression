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

marketing_squared=X_train[:,0]**2
X_poly=np.c_[X_train,marketing_squared]

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
    m=x_train.shape[0]
    total_cost=0
    for i in range(m):
        prediction=np.dot(x_train[i],w)+b
        error=-y_train[i]+prediction
        total_cost+=error**2
    total_cost=(total_cost/(2*m))  
    return total_cost

#function to compute total cost 
def compute_gradient(x_train,y_train,w,b):
    m,n=x_train.shape
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
    m,n=x_train.shape
    cost_history=[]
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(x_train,y_train,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        cost=compute_cost(x_train,y_train,w,b)
        cost_history.append(cost)
    return w,b,cost_history

alpha=0.01
iterations=1000

#training linear model
x_linear,mean_linear,sd_linear=normalization(X_train)
w_linear=np.zeros(X_train.shape[1])
b_linear=0
w_linear,b_linear,cost_linear=gradient_descent(
    x_linear,
    y_train,
    w_linear,
    b_linear,
    iterations,
    alpha
)
linear_final_cost=cost_linear[-1]

#training polynomial model
x_poly,mean_poly,sd_poly=normalization(X_poly)
w_poly=np.zeros(X_poly.shape[1])
b_poly=0
w_poly,b_poly,cost_poly=gradient_descent(
    x_poly,
    y_train,
    w_poly,
    b_poly,
    iterations,
    alpha
)
poly_final_cost=cost_poly[-1]

#comparing the models
print("\n===== MODEL COMPARISON =====")
print("Linear Regression Cost :",linear_final_cost)
print("Polynomial Regression Cost :",poly_final_cost)
print("Linear Model Features:",X_train.shape[1])
print("Polynomial Model Features:",X_poly.shape[1])


if poly_final_cost < linear_final_cost:
    print("\nPolynomial model performs better")
else:
    print("\nLinear model performs better")

#plotting the curve for cost vs iterations
plt.figure()
plt.plot(cost_linear,label="Linear Regression")
plt.plot(cost_poly,label="Polynomial Regression")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.savefig("images/cost_comparison.png")
plt.grid(True)

#plotting spend vs profit
plt.figure()
plt.scatter(X_train[:,0],y_train)
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.title("Marketing spend vs Profit")
plt.savefig("images/marketing_vs_profit.png")
plt.grid(True)


#plotting size vs profit
plt.figure()
plt.scatter(X_train[:,1],y_train)
plt.xlabel("Size")
plt.ylabel("Profit")
plt.title("Team size vs Profit")
plt.savefig("images/team_size_vs_profit.png")
plt.grid(True)


#plotting years active vs profit
plt.figure()
plt.scatter(X_train[:,2],y_train)
plt.xlabel("Years Active")
plt.ylabel("Profit")
plt.title("Years Active vs Profit")
plt.savefig("images/years_active_vs_profit.png")
plt.grid(True)

plt.show()