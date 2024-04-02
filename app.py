''' 
# old code:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')
user_input= st.text_input('Enter Stock Ticker', 'AAPL')
yfin.pdr_override()

spy = pdr.get_data_yahoo(user_input, start='2010-10-24', end='2019-12-23')

#Describing data
st.subheader('Data from 2010 - 2019')
st.write(spy.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(spy.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=spy.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(spy.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=spy.Close.rolling(100).mean()
ma200=spy.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(spy.Close)
st.pyplot(fig)

#splitting data into training and testing

data_training=pd.DataFrame(spy['Close'][0:int(len(spy)*0.70)])
data_testing=pd.DataFrame(spy['Close'][int(len(spy)*0.70):int(len(spy))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

#load the model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_spy = past_100_days.append(data_testing, ignore_index=True)
input_data=scaler.fit_transform(final_spy)


x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


#final visualization
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# old code ends here. 
'''

# To run this app paste this command in the terminal: "streamlit run app.py"

# updated code with fixed bugs:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
yfin.pdr_override()

spy = pdr.get_data_yahoo(user_input, start='2014-01-01', end='2023-12-31') # can be changed to desired time range

# Describing data
st.subheader('Data from 2014 - 2023') 
st.write(spy.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(spy.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = spy.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(spy.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = spy.Close.rolling(100).mean()
ma200 = spy.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(spy.Close)
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(spy['Close'][0:int(len(spy)*0.70)])
data_testing = pd.DataFrame(spy['Close'][int(len(spy)*0.70):int(len(spy))])

# Check if data_training is empty
if not data_training.empty:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load the model
    model = load_model('keras_model.h5')

    # Testing part
    past_100_days = data_training.tail(100)
    # final_spy = past_100_days.append(data_testing, ignore_index=True)
    final_spy = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_spy)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Reshaping x_test to match the input shape expected by the model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predicting using the model
    y_predicted = model.predict(x_test)

    # Scaling back the predicted values
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor

    # Final visualization
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
else:
    st.write("Insufficient data for training. Please choose another stock ticker.")
