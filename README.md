# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HEMAPRASAD N
RegisterNumber:  212222040054
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

df.head()

![ML21](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/389aec67-0fcf-49d8-a704-fe8b2413c4ec)

df.tail()

![ML22](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/d5744e68-76a9-4ff6-8da3-6dc0cbaf7d73)

Array value of X

![ML23](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/240ffb61-08d1-4880-9a4f-dc3602dae807)

Array value of Y

![ML24](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/1a744ad1-b7a9-4d82-8734-5a1da2f38a5e)

Values of Y prediction

![ML25](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/efb13848-011c-41c1-aad8-66b383b623a8)

Array values of Y test

![ML26](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/df2ad429-9245-4dad-adff-9b6ce447396d)

Training Set Graph

![ML27](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/87c95251-fcd4-40ed-ba2e-1baf74e5cbe0)

Test Set Graph

![ML28](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/83c085bd-c17d-4035-9390-3d5cb1626a1e)

Values of MSE, MAE and RMSE

![ML29](https://github.com/Hemaprasad-N/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135933397/9a8da89f-f18b-4e98-be47-d7fa5371e43c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
