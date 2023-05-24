import streamlit as st
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import date
from PIL import Image

@st.cache_data
def get_invested_stocks():
    df = pd.read_excel("Our_Portfolio2.xlsx")
    # st.subheader("Table before removing NaN values:")
    # st.table(df)
    df.dropna(inplace = True)
    # st.subheader("Table after removing NaN values:")
    # st.write(df.columns)
    stock_lst = df["Stock Name"]
    stock_lst = stock_lst+".NS"
    return stock_lst
def plot_me(stock,start,today):
    data = yf.download(stock, start,today)
    data.reset_index(inplace = True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=stock))
    fig.update_layout(title=stock, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

def get_data():
    # Define list of Nifty 50 stock symbols
    #symbols = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS']
    symbols = get_invested_stocks()
    # st.subheader("The stocks which you have invested are:")
    # st.write(symbols)
    # Get historical stock prices for each symbol
    st.write()
    selected = []
    
    today = date.today().strftime("%Y-%m-%d")
    selected.append("^NSEI")
    selected.append("^NSEBANK")
    data = yf.download(selected, start='2023-01-01', end=today)['Close']

    percent_change = data.pct_change()

    # Return the returns DataFrame
    return percent_change
    # return symbols

def create_heatmap(data):
    # Create correlation matrix
    # st.write("data inside heatmap ishere is ")
    # st.write(data)
    corr_matrix = data.corr()

    # Create heatmap
    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)


    ax.set_title('Heatmap')
    ax.set_xlabel('Stock Symbols')
    ax.set_ylabel('Stock Symbols')
    st.pyplot(fig)

def get_our_data(one,two):
    # Define list of Nifty 50 stock symbols
    #symbols = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS']
    
    # Get historical stock prices for each symbol
    st.write()
    selected = []
    
    today = date.today().strftime("%Y-%m-%d")
    selected.append(one)
    selected.append(two)
    data = yf.download(selected, start='2022-01-01', end=today)['Close']

    percent_change = data.pct_change()

    # Return the returns DataFrame
    return percent_change
    # return symbols

def show_market_insights():

    st.header("How is market performing in the recent times?")
    how = Image.open('bse.jpg')
    st.image(how, caption='BSE') 
    st.write("Analyzing how each stock has performed is tough as we have thousands of stocks ...")
    st.write("So let us see how nifty has performed?")
    start = "2017-01-01"
    today = date.today().strftime("%Y-%m-%d")
    plot_me("^NSEI",start,today)
    # nifty = yf.download("^NSEI",start,today) 
    # st.plotly_chart(nifty["Close"])
    # heatmap for nifty
    symbols = get_invested_stocks()
    st.subheader("The stocks which you have invested are:")
    st.write(symbols)
    data2 = get_data()
    data = data2.dropna()
    
    st.write("The heatmap for the data  ",data)
    # nasdaq and it index
    st.write("How correlated are NIfty and BankNifty ?")
    # Creating heatmap
    create_heatmap(data)
    # st.write("Figure created")
    st.write("Let's see the correlation between Nasdaq and ITBEES")
    data3 = get_our_data("^IXIC","^CNXIT")
    data3.dropna(inplace = True)
    st.table(data3.head(10))
    create_heatmap(data3)
    st.header("Why consider correlation?")
    st.write("We are here to explain")

show_market_insights()


