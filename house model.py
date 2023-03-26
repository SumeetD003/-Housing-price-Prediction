import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#loading the dataset 
data = pd.read_csv('Housing.csv') 

#splitting the data into features and target variable 
X = data[['area','bedrooms','price']] #features 
y = data['price'] #target variable 

#splitting the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

 #creating a linear regression model 
regressor = LinearRegression()  

 #fitting the model on the training set 
regressor.fit(X_train, y_train)  

 #predicting on the test set results 
y_pred = regressor.predict(X_test)  

 #evaluating the model performance using mean squared error and r2 score 
mse = mean_squared_error(y_test, y_pred)  

 #printing out the results 
print("Mean squared error: %.2f" % mse)  

 #calculating r2 score of our model on test set results 
r2 = r2_score(y_test, y)