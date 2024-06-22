import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=5)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = BedrockChat(
        client=bedrock,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\\n\\nHuman"],
        },
    )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at the end but summarize with 500 words with detailed explanations.
     Answer based on given context. Answer only if response aligns with the given context. Dont make any irrelevant answer
    <context> {context} </context>
    Question: {question}
    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 30}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Crop Recommender")
    st.header("Get Crop Recommendation with Claude 🧑‍🌾🌾")

    # Input fields for different parameters
    nitrogen_level = st.number_input("Nitrogen Level", min_value=0.0, step=1.0)
    phosphorus_level = st.number_input("Phosphorus Level", min_value=0.0, step=1.0)
    potassium_level = st.number_input("Potassium Level", min_value=0.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph_value = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                st.text(docs)
                get_vector_store(docs)
            st.success("Done")

    if st.button("Get Crop Recommendation"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            user_question = f"Nitrogen level is {nitrogen_level}, Phosphorus level is {phosphorus_level}, Potassium level is {potassium_level}, Temperature is {temperature}°C, Humidity is {humidity}%, pH value is {ph_value}, and Rainfall is {rainfall} mm .Ensure to include a diverse range of plants from the following categories: Grains (e.g., Rice, Maize), Legumes (e.g., ChickPea, KidneyBeans, PigeonPeas, MothBeans, MungBean, Blackgram, Lentil), Fibers (e.g., Jute, Cotton), Beverage Crops (e.g., Coffee), Consider all parameters and ensure that the recommendations are suitable for the given environmental conditions.which crops are recommended?"
            st.write(get_response_llm(llm, faiss_index, user_question))
        st.success("Done")

if __name__ == "__main__":
    main()