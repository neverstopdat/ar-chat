import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, OpenAIEmbeddings  # Assuming custom combined import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from PIL import Image

HN_IMAGE = Image.open("img/hn_logo.png")
st.set_page_config(page_title="NewsNerd HackerBot 🤖📰")
st.title("NewsNerd HackerBot 🤖📰")
stop = False

with st.sidebar:
    st.image(HN_IMAGE)
    st.markdown("""
    # **Greetings, Digital Explorer!**

    Are you fatigued from navigating the expansive digital realm in search of your daily tech tales 
    and hacker happenings? Fear not, for your cyber-savvy companion has descended upon the scene – 
    behold the extraordinary **NewsNerd HackerBot**!
    """)



st.title("Direct Question Answering with OpenAI")

# Ensure the OPENAI_API_KEY is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize the LLM with the specific model and API key
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=OPENAI_API_KEY)

# Section for direct question-answering
st.header("Ask a direct question")
user_query = st.text_input("Enter your question here:", placeholder="Who is Leo demo123? and how old is he?")
if st.button('Submit Direct Question'):
    if user_query:
        with st.spinner('Asking OpenAI...'):
            try:
                response = llm.invoke(user_query)
                if response:
                    st.write("Answer:", response)
                else:
                    st.error("No response found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a question.")

# Section for uploading a PDF and asking questions based on its content
st.header("Upload a PDF and ask a question")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
pdf_query = st.text_input("Enter your question based on the PDF content:", placeholder="What does the document say about XYZ?")
if st.button('Submit PDF Question'):
    if pdf_query and uploaded_file:
        with st.spinner('Processing PDF...'):
            try:
                reader = PdfReader(uploaded_file)
                combined_text = ''.join([page.extract_text() or '' for page in reader.pages])
                
                if not combined_text:
                    st.error("Failed to extract text from PDF.")
                    st.stop()

                text_splitter = CharacterTextSplitter()
                finalData = text_splitter.split_text(combined_text)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                documentsearch = FAISS.from_texts(finalData, embeddings)
                chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff")

                docs = documentsearch.similarity_search(pdf_query)
                response = chain.invoke(input={'input_documents': docs, 'question': pdf_query})

                if response:
                    st.write("Answer:", response['output_text'])
                else:
                    st.error("No response found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a PDF file and enter a question.")
