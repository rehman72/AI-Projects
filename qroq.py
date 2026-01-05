import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
#load the Groq API Key

load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only
    Please Provide the most accurate response based on the question
    <context>
    {context}
    Questions:{input}
"""
)

document_chain=create_stuff_documents_chain(llm,prompt)

retriever=st.session_state.vectors.as_retriever()
retriever_chain=create_retrieval_chain(retriever=retriever,combine_docs_chain=document_chain)

prompt=st.text_input("Input Your Prompt Here...")

if prompt:
    start=time.process_time()
    response=retriever_chain.invoke({"input": prompt})
    print("Response Time: ",time.process_time()-start)
    st.write(response['answer'])


    with st.expander("Document Similarity Search"):
        #Find the Relevant Chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------")

    






