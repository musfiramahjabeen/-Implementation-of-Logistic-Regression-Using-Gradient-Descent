# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and Load the Dataset

2.Drop Irrelevant Columns (sl_no, salary)

3.Convert Categorical Columns to Category Data Type

4.Encode Categorical Columns as Numeric Codes

5.Split Dataset into Features (X) and Target (Y)

6.Initialize Model Parameters (theta) Randomly

7.Define Sigmoid Activation Function

8.Define Logistic Loss Function (Binary Cross-Entropy)

9.Implement Gradient Descent to Minimize Loss

10.Train the Model by Updating theta Iteratively

11.Define Prediction Function Using Threshold (0.5)

12.Predict Outcomes for Training Set

13.Calculate and Display Accuracy

14.Make Predictions on New Data Samples
## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MUSFIRA MAHJABEEN M
RegisterNumber:  212223230130

import pandas as pd
import numpy as np
df=pd.read_csv("Placement_Data.csv")
df
df=df.drop('sl_no',axis=1)
df=df.drop('salary',axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))

def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

*/
```

## Output:
### Value of df
![image](https://github.com/user-attachments/assets/52b493ff-5d25-46dd-a5c5-933a414bd5c6)

### df.head()
![Screenshot 2025-03-29 193054](https://github.com/user-attachments/assets/14e4ce38-2ce3-4e76-9cea-d6083ab46193)

### Value of df.dtypes
![Screenshot 2025-03-29 193101](https://github.com/user-attachments/assets/6219bc7b-a0e6-46f2-a825-8b9bdb67734c)

### Value of df
![Screenshot 2025-03-29 193111](https://github.com/user-attachments/assets/2c435072-1df7-4413-a77d-f083827f05cf)

### Value of y
![Screenshot 2025-03-29 193120](https://github.com/user-attachments/assets/91adca36-a374-4815-8c6d-11f5a0a8deaa)

### Value of accuracy 
![Screenshot 2025-03-29 193137](https://github.com/user-attachments/assets/2a09ca59-1b63-4c98-98a2-da3a9d8d6446)

### Value of y_prednew
![image](https://github.com/user-attachments/assets/06ad32fc-2d63-46fb-8514-e9a60e673433)


![image](https://github.com/user-attachments/assets/4d13b10d-7817-4e3a-b2d5-82beb9a1df15)












## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
