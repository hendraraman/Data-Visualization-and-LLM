import streamlit as st
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from PIL import Image
import plotly.graph_objs as go
# ARIMA model
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.metrics import *
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from PIL import Image
import tensorflow



@st.cache_data


def rsi_ma_adder(df1):
    df1['MA50'] = df1['Close'].rolling(window=50).mean()

    # we are calculating rsa 
    delta = df1['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df1['RSI'] = 100 - (100 / (1 + rs))
    return df1

def plot_me_and_return(stock,start,today):
    if stock == "BAJAJFINANCE":
        stock = "BAJFINANCE"
    stock = stock+".NS"
    data = yf.download(stock, start,today)
    # data.reset_index(inplace = True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[:-150], y=data['Close'].iloc[:-150], name=stock))
    fig.update_layout(title=stock, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    return data



def accuracy_finder(data):

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    n_steps = 10
    st.header("Metrics for LSTM: ")
    image = Image.open('lstm.png')

    st.image(image, caption='LSTM Image', use_column_width=True)

    X = []
    y = []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i - n_steps:i])
        y.append(data_scaled[i, 0])

    # X = np.array(X)
    # y = np.array(y)
    X = np.reshape(X, (len(X), 40))
    # X = pd.DataFrame(X)
    # y = pd.DataFrame(y)
    st.write("X is ",X.shape)
    # Split the data into training and testing sets
    # train_size = int(0.8 * len(X))
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X_train = X.iloc[:-100,:].values
    X_test = X.iloc[-100:,:].values
    y_train = y.iloc[:-100,0].values
    y_test = y.iloc[-100:,0].values

    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)
    # y_train = pd.DataFrame(y_train)
    # y_test = pd.DataFrame(y_test)

    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # Reshape the input data to be 3-dimensional (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], 10,  4))
    X_test = X_test.reshape((X_test.shape[0], 10,  4))


    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 4)))  # Update the input shape
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Evaluate the model on the testing data
    loss = model.evaluate(X_test, y_test, verbose=0)


    # st.write('Test Loss:', loss)
    # Evaluate the model on the testing data
    y_pred = model.predict(X_test)

    # Calculate mean squared error (MSE)
    mse = np.mean((y_test - y_pred)**2) 
    # st.write("The RMSE value is ",rmse)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R-squared:", r2)


    # SVR Model:

    # Reshape the data to have two dimensions
    st.header("For SVR model: ")
    image = Image.open('svr.jpg')

    # Display the image using Streamlit
    st.image(image, caption='Support Vector Regressor', use_column_width=True)
    
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    y_train_2d = y_train.reshape(y_train.shape[0])

    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_test_2d = y_test.reshape(y_test.shape[0])

    # Create and train the SVR model
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_2d, y_train_2d)

    # Make predictions
    y_pred = svr_model.predict(X_test_2d)
    # new ends

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R-squared:", r2)

    # ARIMA
    st.header("Arima Model: ")
    image = Image.open('arima2.png')

    # Display the image using Streamlit
    st.image(image, caption='Arima Image', use_column_width=True)
    model = ARIMA(y_train,order = (1,0,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps = len(X_test))

    # Evaluate model performance
    mse = mean_squared_error(y_test, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, forecast)

    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R-squared Score:",r2)

    # random forest regressor 
    st.header("Random Forest Regressor: ")

    image = Image.open('random.png')
    st.image(image, caption='Random Forest Regressor', use_column_width=True)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_2d,y_train)
    y_pred = model.predict(X_test_2d)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("R-squared Score:",r2)




df = pd.read_excel("Our_Portfolio2.xlsx")
    # st.subheader("Table before removing NaN values:")
    # st.table(df)
df.dropna(inplace = True)
# st.subheader("Table after removing NaN values:")
# st.write(df.columns)
stock_lst = df["Stock Name"]
# st.write(stock_lst)

stock_lst = tuple(stock_lst)
selected_stock = st.selectbox("Select the stock you want to find acccuracy for : ",stock_lst)
start = "2022-05-01"

today = date.today().strftime("%Y-%m-%d")
data = plot_me_and_return(selected_stock,start,today)
# st.write("returned data ")
# st.write(data)
# data.reset_index(inplace = True)
data = data.iloc[:,[3,5]]
data = rsi_ma_adder(data)
# removing first 50 rows
data = data.iloc[50:,:]
st.write("Features used for Regression-> we newly created -->RSI and MA ",data)
# test_data = data.loc[-150:,[3,5]]
accuracy_finder(data)
