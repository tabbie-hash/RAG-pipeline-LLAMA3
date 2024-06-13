# RAG-pipeline-LLAMA3

## Objective

Use Llama3, Langchain and FAISS to create a Retrieval Augmented Generation (RAG) system. This will allow us to ask questions about our ICD10 document (that was not included in the training data), without fine-tunning the Large Language Model (LLM).
When using RAG, if you are given a question, you first do a retrieval step to fetch any relevant documents from a special database, a vector database where these documents were indexed. 

## Definitions

* LLM - Large Language Model  
* Llama3 - LLM from Meta 
* Langchain - a framework designed to simplify the creation of applications using LLMs
* Vector database - a database that organizes data through high-dimmensional vectors  
* FAISS - vector database  
* RAG - Retrieval Augmented Generation (see below more details about RAGs)


## What is a Retrieval Augmented Generation (RAG) system?

Large Language Models (LLMs) has proven their ability to understand context and provide accurate answers to various NLP tasks, including summarization, Q&A, when prompted. While being able to provide very good answers to questions about information that they were trained with, they tend to hallucinate when the topic is about information that they do "not know", i.e. was not included in their training data. Retrieval Augmented Generation combines external resources with LLMs. The main two components of a RAG are therefore a retriever and a generator.  
 
The retriever part can be described as a system that is able to encode our data so that can be easily retrieved the relevant parts of it upon queriying it. The encoding is done using text embeddings, i.e. a model trained to create a vector representation of the information. The best option for implementing a retriever is a vector database. As vector database, there are multiple options, both open source or commercial products. Few examples are ChromaDB, Mevius, FAISS, Pinecone, Weaviate. Our option in this Notebook will be a local instance of ChromaDB (persistent).
