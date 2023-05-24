
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


# @st.cache_data

def main():
    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))
    # st.set_page_config(page_title="MY_GPT FOR PDF's")
    st.header('Ask your pdf')

    # uploaded pdf
    pdf = st.file_uploader("Upload here ",type = "pdf")
    
    # extracting text from pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        #splitting into chunks using langchains CharacterTextSplitter object
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)

        # creating embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings )
        ans = st.text_input("Do you wanna continue? true or false")
        # while(True):

        # question for gpt from pdf
        user_question = st.text_input("Ask ur question")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)


            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            reply = chain.run(input_documents = docs, question = user_question)
            st.write("Reply is ",reply)
                




if __name__ == '__main__':
    main()