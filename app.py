import streamlit as st
import pandas as pd
import requests
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI, ChatOpenAI

# Streamlit app title
st.title("Compliance Measures QA System")

# Function to fetch data from the API
# Load data and create index
loader = CSVLoader(file_path='db.csv')
data = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key="OPEN_API_KEY")
index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader])
retriever = docsearch.vectorstore.as_retriever()

llm = ChatOpenAI(openai_api_key="OPEN_API_KEY", model_name="gpt-3.5-turbo", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# Create question-answering chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, input_key="question")

# User input for the query
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    # Pass the query to the chain
    response = chain({"question": query})
    st.write(response['result'])

# Display the data fetched
#st.write("Data Fetched from API:")
