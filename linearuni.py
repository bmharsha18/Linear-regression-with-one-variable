"""
Linear regression with one variable

Suppose you are the CEO of a
restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.

You would like to use this data to help you select which city to expand
to next.

The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss.

"""
import csv
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import numpy as np

def compute_cost(X,y,theta):
    m = len(y)
    J = 0
    hx = list()
    squarederrors = list()
    for i in X: 
        hofx = theta[0]*i[0] + theta[1]*i[1]
        hx.append(hofx)
    for i in range(m):
        temp = float(hx[i]) - float(y[i])
        result = temp*temp
        squarederrors.append(result)
    J = (1/(2*m))*sum(squarederrors)
    J = np.array(J)
    return J    
        

def gradient_descent(X,y,theta,alpha,iterations):
    m = len(y)
    J_old_values = list()
    J_old_values.append(compute_cost(X,y,theta))
    for i in range(iterations):
        hofx = list()
        errorvalues = list()
        sum1 = 0
        for i in X:
            temp1 =  theta[0]*i[0] + theta[1]*i[1]
            hofx.append(temp1)
        for i in range(m):
            error = float(hofx[i]) - float(y[i])
            errorvalues.append(error)
        theta0 = theta[0] - ((alpha/m)*sum(errorvalues))
        for i in range(len(X)):
            sum1+=(errorvalues[i]*X[i][1])
            temp2 = (alpha/m)*sum1
        theta1 = theta[1] - temp2
        theta = [theta0,theta1]
        J_old_values.append(compute_cost(X,y,theta))
    return theta,J_old_values

#reading the data from dataset.csv
print('\nLinear regression with one variable to predict profits for a food truck.')
dataset = list()
fp=open('dataset.csv','r')
reader = csv.reader(fp , delimiter=',')
for row in reader:
    dataset.append(row)   
m=len(dataset)
print('No number training samples : ',m) 
print('------------Reading dataset values--------------')
xvalues = list()
X = list()
y = list()
    
for i in range(m):
    X.append([1])

index = 0    
for i in dataset:
    X[index].append(float(i[0]))
    xvalues.append(i[0])
    y.append(i[1])
    index+=1

#plotting the given dataset values

print('------------Plotting the given dataset values--------------')
xmin = float(min(xvalues)) - 6
xmax = float(max(xvalues)) + 16

ymin = float(min(y)) - 5
ymax = float(max(y)) + 18
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.axis([xmin,xmax,ymin,ymax])
plt.plot(xvalues,y,'rx')
plt.show()

temp1 = ([0],[0])
theta = np.array(temp1)
Xmat = np.array(X) 

no_of_iterations = 1500
alpha = 1e-05

theta,J_cost = gradient_descent(X, y, theta, alpha, no_of_iterations)
print("\n-------------computing gradient_descent------------------")
print("\ntheta values are: \n",theta[0],"\n",theta[1])

print('------------Fitting the most suitable line through the samples--------------')
plt2.ylabel('Profit in $10,000s')
plt2.xlabel('Population of City in 10,000s')
plt2.axis([xmin,xmax,ymin,ymax])
plt2.plot(xvalues,y,'rx')
x_values = np.array(plt.gca().get_xlim())
y_values = theta[0] + theta[1] * x_values
plt2.plot(x_values, y_values, '--')    
plt2.show()

print('------------Plotting cost function J verses number of iterations--------------')
plotx = list()
ploty = list()
for i in range(len(J_cost)):
    plotx.append(i)
for i in J_cost:
    ploty.append(i)
plt3.xlabel('No of iterations')
plt3.ylabel('Cost function "J" ')
plt3.plot(plotx,ploty,'b')
plt3.show()

#To predict values [1,3.5] and [1,7]
while(True):
    input_pop = float(input("\nEnter the Population value: (Scale in 10,000's)\n"))
    predict = theta[0] + input_pop * theta[1]
    predict = predict * 10000
    print("Profit predicted in the area of population",int(input_pop*10000),"=",round(float(predict),2),"$")   
    abc = int(input("-1 to exit , 5 to continue with other population value!\n"))
    if(abc==-1):
        break




    




    
    
