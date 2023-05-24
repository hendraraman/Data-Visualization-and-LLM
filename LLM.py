import streamlit as st
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import date
from PIL import Image

# chat messages
from langchain.chat_models import ChatOpenAI
from langchain.schema import *
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate

@st.cache_data
# current author's books

def books_of(author):
    if author.lower() == "ray dalio":
        return "Principles by ray dalio"
    elif author.lower() == "charlie munger":
        return "Poor Charlie's Almanack"
    elif author.lower() == "robert toru kiyosaki":
        return "Rich Dad Poor Dad"
    elif author.lower() == "napoleon hill":
        return "Think and grow rich"
    
def show_llm_page():
    # current_author = st.text_input(" Enter the author's name you want GPT to be: ")
    
    current_author = st.selectbox("Select a famous person below whose advice you want to get? ",("ray dalio","charlie munger","robert toru kiyosaki","napoleon hill"))
    current_book = books_of(current_author)
    prompt = PromptTemplate(
        input_variables=["author","books"],
        template="""Act like you are {author}. he is the author of the books {books}. 
        I want you to give financial tips as {author} would give based on the books. 
        Read his books carefully and give tips based on them""",
    )
    st.write("Created prompt is ",prompt.format(author=current_author,books = current_book   ))
    new_prompt = prompt.format(author=current_author,books = books_of(current_author))
    # prompt.format(author=current_author,books = books_of(current_author))
    st.write(new_prompt)
    try:
        author_img = Image.open(current_author+'.jpg')
        st.image(author_img, caption=current_author)
    except:
        pass
    st.write("Advice loading 🔃🏃🏃‍♂️🏃‍♀️ ")
    answer = llm(new_prompt)
    st.header("Here is what "+current_author+" has to say to you...")
    st.write(answer)
    




llm = OpenAI(temperature=1)

show_llm_page()
