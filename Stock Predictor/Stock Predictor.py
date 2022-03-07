#Description: This program uses an artificial recurrent neural network called Long Short Term memory(LSTM) to predict the closing stock price
#of a corporation using the past 60 days

#Import Libraries
import math
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yfin
yfin.pdr_override()

#################################################################################################################

#Get the stock quote
df = pdr.DataReader('DJI', data_source = 'yahoo', start='2016-06-04', end= '2021-06-04') #5 years worth of data exported directly from yahoo finance 
#Show the data
print(df)

# Create a Pandas dataframe from the data.
df = pd.DataFrame({'Open': df['Open'], 'High': df['High'], 'Low': df['Low'], 'Close': df['Close']})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandas_simplee.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

#Get the number of rows and columns in the data set
df.shape #(rows, columns)
#Visualize the closing price history on a chart
plt.figure(figsize = (16,8)) #size of the chart 
plt.title('Close Price History')
plt.plot(df['Close'], color = 'green')
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Close Price USD ($)', fontsize = 16)
plt.show()


#Create a new dataframe with only the 'Close column'
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)# Train it with 80% of the data that we have. Used math.ceil to round up the length training set
print(training_data_len)


#Scale the data
'''
Apply preprocessing transformations scaling or
normalization to the input data before it is presented to a neural network.
Applied Gradient discent...............................................................
'''
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :] #Contains values from 0 to training_data_len and (, :]) gets back all of the columns

#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)): # Append the last 60 values to our x_train dataset
    x_train.append(train_data[i - 60:i, 0]) #x_train will contain 60 values. Indexes from position 0 to 59.
    y_train.append(train_data[i,0]) #y_train will contain index 60 
    if i <= 60:
        print(x_train) #print past 60 values
        print(y_train) # contains the 60 first value that we want our model to predict
        print()

#Convert the x_train and y_train to numpy arrays so we can use them to train the LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)    

#Reshape the Data (make input 3 dimensional) because the LSTM expects the input to be 3 dimensional
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1)) # we need to input the number of samples(number of rows that we have),
#number of time steps(number of columns) and number of features(just the closing price)
x_train.shape

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))#Add the LSTM layer with 50 neurons
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25)) #Add a dense layer 
model.add(Dense(1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error') #Optimizer is used to improve upon the loss function and the loss function measures how well the model did

#Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1) #batch size: total number of training examples in a single batch, epochs number of iterations when an entire dataset is past forward

#Create the testing dataset
#Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: , :]

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
#Convert the data to a numpy array
x_test = np.array(x_test)


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # we need to input the number of samples(number of rows that we have),
#number of time steps(number of columns) and number of features(just the closing price)

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #unscaling the values

# disable chained assignments
pd.options.mode.chained_assignment = None

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Close Price USD ($)', fontsize = 16)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()

#Show the valid and predicted prices
print(valid)

#Get the root mean squared error (RMSE) to evaluate the model 
rmse = np.sqrt(np.mean(predictions - y_test)**2 )
print("RMSE on test data: " + str(rmse))

#Get the R squared 
pred = np.array(predictions)
print("R squared on test data: " + str(r2_score(y_test, pred)))
#print("R squared on test data: " + str(mean_squared_error(y_test, pred)))

########################################################################

#Get the quote
nio_quote = pdr.DataReader('DJI', data_source = 'yahoo', start='2016-06-04', end= '2021-06-04')

#Create a new dataframe
new_df = nio_quote.filter(['Close'])

#Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Create an empty list
X_test = []

#Append the past 60 days
X_test.append(last_60_days_scaled)

#Convert the X_test data set to a numpy array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price
pred_price = model.predict(X_test)

#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#Get the quote 
nio_quote2 = pdr.DataReader('DJI', data_source = 'yahoo', start = '2021-06-04', end = '2021-06-04')
print(nio_quote2['Close'])


