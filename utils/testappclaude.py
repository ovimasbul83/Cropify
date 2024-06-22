import json
import os
import sys
import boto3
import streamlit as st
import pandas as pd

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=20)
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
            "temperature": 0.2,  # Adjusted temperature for more variability
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\\n\\nHuman"],
        },
    )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at the end but summarize with detailed explanations.
    Consider a wide variety of plants and ensure diversity in your recommendations.
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
            search_type="similarity", search_kwargs={"k": 35}  # Increased k for more variety
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    # Load the test set
    test_set_file_path = 'Crops_Recommendation_test_set.csv'
    test_set = pd.read_csv(test_set_file_path)

    # Process each row in the test set
    results = []

    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_claude_llm()

    for index, row in test_set.iterrows():
        nitrogen_level = row['Nitrogen']
        phosphorus_level = row['Phosphorus']
        potassium_level = row['Potassium']
        temperature = row['Temperature']
        humidity = row['Humidity']
        ph_value = row['pH_Value']
        rainfall = row['Rainfall']

        user_question =f"Nitrogen level is {nitrogen_level}, Phosphorus level is {phosphorus_level}, Potassium level is {potassium_level}, Temperature is {temperature}°C, Humidity is {humidity}%, pH value is {ph_value}, and Rainfall is {rainfall} mm .Ensure to include a diverse range of plants from the following categories: Grains (e.g., Rice, Maize), Legumes (e.g., ChickPea, KidneyBeans, PigeonPeas, MothBeans, MungBean, Blackgram, Lentil), Fibers (e.g., Jute, Cotton), Beverage Crops (e.g., Coffee), Consider all parameters and ensure that the recommendations are suitable for the given environmental conditions.which crops are recommended?"

        response = get_response_llm(llm, faiss_index, user_question)
        results.append({'Query': user_question, 'Response': response})

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_file_path = 'Crop_Recommendation_Responses.csv'
    results_df.to_csv(results_file_path, index=False)

    print("Responses have been saved to:", results_file_path)

if __name__ == "__main__":
    main()
