# Stock-Predictor-
Stock Predictor using Long Short-Term Memory (LSTM)

Forecasting is the process of predicting the future based on present and historical facts.
The main issue is to understand the patterns in the data series and then use this pattern to forecast the future.
Hand-coding the patterns would be time-consuming and would require revisions for the following data set.
Deep Learning has proved to be more effective at recognizing patterns in both organized and unstructured data.
We need networks to analyze patterns across time in order to grasp the patterns in a long series of data.
Recurrent Networks are commonly used for learning such data.
They can recognize long-term and short-term interdependence as well as temporal disparities.
This project was created using the LSTM model which is a kind of RNN architecture.

Steps taken:
1- Imported the data from Yahoo Finance and stored it in an excel sheet using pandas.
2- Created a dataframe using the closing Price of the Stock or Index. 
3- Converted the dataframe to a numpy array and used 80% of the data to train the model
4- Scale the data using Gradient Discent before it is presented to a neural network.
5- Create a Training set and split the data into x_train and y_train 
   X_train: Includes all independent variables,these will be used to train the model
   y_train: This is your dependent variable which needs to be predicted by this model, this includes category labels against your independent variables, we need to specify our        dependent variable while training/fitting the model.
   x_test: The remaining portion of the independent variables from the data which will not be used in the training phase and will be used to make predictions to test the accuracy    of the model.
   y_test: The data has category labels for the test data, these labels will be used to test the accuracy between actual and predicted categories.
6- Convert the x_train and y_train to numpy arrays so we can use them to train the the LSTM model.
7- Make the data (imput) 3 dimentional. we need to input the number of samples (number of rows that we have), number of time steps (number of columns) and number of features (just the closing price).
8- Build the LSTM model and compile it.
9- Train the model.
10- Create the Test set (x_test and y_test).
11- Convert the data to a numpy array and reshape the data into a 3 dimentional input.
12- Get the model's predicted price values.
13- Plot and visualize the data using a line chart.
14- Show the valid and predicted prices.
15- Get the RMSE to evaluate the model.
16- Get the R squared.
18- Get the closing price of the last 60 days and convert the dataframe to an array and scale it between 0 and 1.
19- Append the data into the x_test and reshape it. 
20- Predict the price using model.predict and then undo the scaling.



