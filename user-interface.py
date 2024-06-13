import gradio as gr
import torch
import ollama
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import numpy as np
import pandas as pd
import csv
import re

# Function to load, split, and retrieve documents
def load_and_retrieve_lines(doc):
    with open(doc, 'r') as file:
        lines = file.readlines()

    # Process each line and format the output
    formatted_lines = []
    # Extract descriptions
    descriptions = []
    pattern = re.compile(r'^([A-Za-z0-9]+)\s+(.*)')

    for line in lines:
        match = pattern.match(line)
        if match:
            code = match.group(1)
            description = match.group(2)
            formatted_line = f"{code} is the icd code for {description.strip()}"
            formatted_lines.append(formatted_line)
            descriptions.append(description.strip())

    # Write the formatted output to a new file
    with open('icd10-document.txt', 'w') as file:
        for formatted_line in formatted_lines:
            file.write(formatted_line + '\n')

    with open("./icd10-document.txt") as f:
        each_icd10 = f.read()
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=1, chunk_overlap=0)
        loader = TextLoader('./icd10-document.txt')
        splits = loader.load_and_split(text_splitter)
        embeddings = OllamaEmbeddings(model="llama3")
        # vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        # vectorstore.save_local("faiss_index")
        saved_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return saved_db.as_retriever()

# Function to format documents
def format_lines(lines):
    return "\n".join(line.page_content for line in lines)

# Function that defines the RAG chain
def rag_chain(doc, question):
    retriever = load_and_retrieve_lines(doc)
    retrieved_lines = retriever.invoke(question)
    formatted_context = format_lines(retrieved_lines)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Gradio interface
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text", "text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter the icd10 text file and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch(share=True)