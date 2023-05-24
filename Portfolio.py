import streamlit as st
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from PIL import Image


@st.cache_data



def show_me_alerts(user,total_amt_invested):    # shows alert if invested in a single stock for more than 5%
        user.dropna(inplace = True )

        stock_name = user["Invested"]
        st.subheader("Alerts 💀⚡")
        st.write("You have invested more than 5 percent in these stocks")
        more_than_5 = []                            # boolean arrays stores if invested amt is more than 5%
        for i in user.loc[:,"Invested"]:
            try:
                if i/total_amt_invested*100 > 5:
                    more_than_5.append(True)
                    # st.write("Iffff")
                    st.bar_chart(i,i/total_amt_invested*100)
                    #st.write("Stock is invested for more than 5% ")
                else:
                    more_than_5.append(False)
            except:
                st.write()
        iter = 0
        
        for i in user.loc[:,"Stock Name"]:
            try:
                if more_than_5[iter]:
                    if user.loc[iter,"Sector"] == "DIVERSIFIED": #ignores if the stock or ETF is diversified 
                        # st.write("Passed ",i)
                        pass
                    else:
                        st.write(i ," is invested for more than 5%")
                        
                else:
                    pass
                    # st.write("Less than 5 ",i)

                    # if user.loc["Sector"].iloc[iter] == "Diversified":
                    #     pass
                    # else:
                    #     
                iter += 1

                # st.write(total_amt_invested)
            except:
                st.write()
        st.header("Does that Matter? ")
    

def show_portfolio_page():
    # this is where data visualization is performed

    st.header("This is your portfolio")
    
    df = pd.read_excel("Our_Portfolio2.xlsx")
    st.subheader("Table before removing NaN values:")
    st.table(df)
    df.dropna(inplace = True)
    st.subheader("Table after removing NaN values:")
    st.table(df)
    st.subheader("Here is the visualization of invested amount (in percentages)\n")
    fig1, ax1 = plt.subplots()
    colors = sns.color_palette('pastel')[0:5]
    ax1.pie(df.loc[:,"Invested"],labels=df.loc[:,"Stock Name"],autopct='%.0f%%')
    ax1.axis('equal')
    ax1.set_title("Invested Stocks and its %\n")
    st.pyplot(fig1)    
    st.write("This is the most valuable part") 
    show_me_alerts(df,sum(df["Invested"]))
    st.subheader("The key to make money in the Long run is STAY DIVERSIFIED ")


    
st.title("Portfolio page")
show_portfolio_page()