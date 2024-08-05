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
"""
def get_table(x, limit=100):
    url = f"https://nayaone-compass.bubbleapps.io/version-test/api/1.1/obj/{x}"
    headers = {
        'Authorization': 'YOUR_API_KEY'
    }
    
    all_results = []
    cursor = 0
    
    while True:
        params = {
            'cursor': cursor,
            'limit': limit
        }
        
        response = requests.get(url, headers=headers, params=params)
        response_data = response.json()
        
        results = response_data["response"]["results"]
        remaining = response_data["response"]["remaining"]
        
        all_results.extend(results)
        
        if remaining <= 0:
            break
        
        cursor += limit
    
    return pd.DataFrame(all_results)

# Example usage to fetch data and save to CSV
if 'data' not in st.session_state:
    st.session_state['data'] = get_table('Vendor', limit=100)

df = st.session_state['data']

# Save to CSV if needed
df.to_csv('db.csv', index=False)
"""
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
query = st.text_input("Enter your query:", "In which outcome category can this be put: Strengthen compliance measures to meet regulatory requirements.")

if st.button("Get Answer"):
    # Pass the query to the chain
    response = chain({"question": query})
    st.write(response['result'])

# Display the data fetched
st.write("Data Fetched from API:")
st.dataframe(df)
