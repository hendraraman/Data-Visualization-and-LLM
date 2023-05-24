import streamlit as st
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from PIL import Image

@st.cache_data

def show_welcome_page():
    st.header("Welcome Page")
    retail = Image.open('retail.jpg')
    st.image(retail, caption='Retail investors')   
    st.write("""Retail investors' participation has increased from 2020
        But now, it is decreasing .... Why?
        The market has remained sideways for a long time...
        Retailers have lost interest in market... They say ...We are here to help the retailers
             """) 
    st.subheader("Our Goal here is to educate retail investors so that they could make sensible decisions.")

def get_invested_stocks():
    df = pd.read_excel("Our_Portfolio2.xlsx")
    # st.subheader("Table before removing NaN values:")
    # st.table(df)
    df.dropna(inplace = True)
    # st.subheader("Table after removing NaN values:")
    st.write(df.columns)
    stock_lst = df["Stock Name"]
    stock_lst = stock_lst+".NS"
    return stock_lst

def show_market_insights():
    # heatmap for nifty
    data = get_data()
    # Create heatmap
    fig = create_heatmap(data)

    # Display the heatmap in the Streamlit app
    st.pyplot(fig)

def show_portfolio_page():
    # this is where data visualization is performed

    st.header("This is your portfolio")
    
    df = pd.read_excel("Our_Portfolio.xlsx")
    st.subheader("Table before removing NaN values:")
    st.table(df)
    df.dropna(inplace = True)
    st.subheader("Table after removing NaN values:")
    st.table(df)
    
    
    fig1, ax1 = plt.subplots()
    colors = sns.color_palette('pastel')[0:5]
    ax1.pie(df.loc[:,"Invested"],labels=df.loc[:,"Stock Name"],autopct='%.0f%%')
    ax1.axis('equal')
    ax1.set_title("Invested Stocks and its %\n")
    st.pyplot(fig1)

def get_data():
    # Define list of Nifty 50 stock symbols
    #symbols = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TATACONSUM.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS']
    symbols = get_invested_stocks()
    st.write(symbols)
    # Get historical stock prices for each symbol
    st.write()
    selected = []
    option1 = st.selectbox("Select the symbols for which we want to find the correlation between",symbols)
    selected.append(option1)
    option2 = st.selectbox("Selecr the 2nd symbol",symbols)
    selected.append(option2)
    today = date.today().strftime("%Y-%m-%d")
    data = yf.download(selected, start='2023-01-01', end=today)['Close']

    # # Calculate daily returns for each stock
    percent_change = data.pct_change()

    # Return the returns DataFrame
    return percent_change
    # return symbols

def create_heatmap(data):
    # Create correlation matrix
    st.write("data here is ")
    st.write(data)
    corr_matrix = data.corr()

    # Create heatmap
    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)

    # Add title and axis labels
    ax.set_title('Nifty 50 Heatmap')
    ax.set_xlabel('Stock Symbol')
    ax.set_ylabel('Stock Symbol')

    # Return the heatmap figure
    data = get_data()
    # Create heatmap
    fig = create_heatmap(data)

    # Display the heatmap in the Streamlit app
    st.pyplot(fig)

    

# show_welcome = st.button("Show me the welcome page")
# show_portfolio = st.button("Show me my portfolio")
# market_insight = st.button("Show me the Market insights")
# st.write("All buttons are ",show_welcome,show_portfolio)
st.sidebar.success("Select a page above.")

show_welcome_page()

# if show_portfolio:
#     show_portfolio_page()

# if market_insight:
#     show_market_insights()
