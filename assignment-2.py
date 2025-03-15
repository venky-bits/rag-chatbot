# Standard library imports
import os
import re
import textwrap

# Third-party libraries
import chromadb
import pdfplumber
import torch
import streamlit as st
from streamlit_chat import message  # Chat UI
from huggingface_hub import login
from prompt_toolkit import prompt
from sentence_transformers import CrossEncoder, SentenceTransformer

# LangChain and related imports (should be last)
from langchain_community.llms import Ollama
import ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Basic & Advanced RAG implementation 
# ------------------------------------
# The code implements a Retrieval-Augmented Generation (RAG) system using ChromaDB for document storage 
# and Hugging Face models for reranking.
#
# Text Extraction & Cleaning: It extracts financial reports from PDFs using pdfplumber, cleans them by 
# removing page numbers and extra spaces, and splits them into chunks for indexing.
#
# Embedding & Storage: The SentenceTransformer model generates embeddings for each chunk, which are 
# stored in ChromaDB under a collection named "financial_reports".
#
# Retrieval: When a user submits a query, its embedding is generated and compared with stored embeddings 
# in ChromaDB to retrieve the top-k most relevant chunks based on cosine similarity.
#
# Reranking (Advanced RAG): The retrieved documents are then reranked using a CrossEncoder model 
# (msmarco-MiniLM-L6-en-de-v1), which scores query-document pairs for relevance.
#
# Normalization & Selection: The reranker’s scores are normalized between 0 and 1, then the top-reranked 
# results (default: 3) are selected for final context.
#
# LLM Query Execution: The refined context is passed to an Ollama-based LLM via a prompt template, ensuring 
# it answers only from the retrieved content to minimize hallucination.
#
# Guard Rails & Confidence Score: The response is embedded and compared with the context using cosine similarity; 
# if the score is below 0.55, the response is blocked.
#
# Chat History & UI: Responses, queries, and confidence scores are stored in st.session_state and displayed 
# in Streamlit Chat UI for interactive responses.
#
# Final Output: The AI provides the retrieved answer along with a confidence score, calculated as the average 
# score of the top reranked documents.

# Load a sentence-transformers model for financial text
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Function to extract text from PDF files
def extract_text_from_pdf():
    source_1 = "data/Infosys_financial_report_2023-24.pdf"
    source_2 = "data/Infosys_financial_report_2022-23.pdf"
    pdf_path_1 = os.path.abspath(source_1)
    pdf_path_2 = os.path.abspath(source_2)

    text = ""
    with pdfplumber.open(pdf_path_1) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    with pdfplumber.open(pdf_path_2) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    return text

# Function to clean the extracted text
def clean_text(text):
    # Remove extra spaces, newlines, and page numbers
    text = re.sub(r'\n+', '\n', text)  # Normalize newlines
    text = re.sub(r'Page \d+', '', text)  # Remove page numbers
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.strip()
    return text

# Function to chunk the cleaned text
def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, width=chunk_size)

# Function to get or create a ChromaDB collection
def getCollection():
    chroma_client = chromadb.PersistentClient(path="./data/financial_db")
    collection = chroma_client.get_or_create_collection(name="financial_reports")
    return collection

# Function to embed text chunks and store them in ChromaDB
def embed_text_chunks(chunks):
    chroma_client = chromadb.PersistentClient(path="./data/financial_db")
    
    # Drop the existing collection if it exists
    if "financial_reports" in chroma_client.list_collections():
        chroma_client.delete_collection(name="financial_reports")

    # Create a new collection
    collection = chroma_client.get_or_create_collection(name="financial_reports")

    # Add text and embeddings to ChromaDB
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()  # Convert to list
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"chunk_id": i, "source": "Infosys 2023-24"}],
            ids=[f"chunk_{i}"]
        )
    print("Financial report embeddings stored successfully!")

    return collection

# Function to create embeddings for the financial reports
def create_embeddings():
    text = extract_text_from_pdf()
    cleaned_text = clean_text(text)
    # Chunk the text
    chunks = chunk_text(cleaned_text, chunk_size=300)
    print(f"Total Chunks: {len(chunks)}")
    collection = embed_text_chunks(chunks)
    return collection
    
# Function to initialize the prompt LLM chain
def initPromptLLM_old():
    # Define the prompt template
    template = """
    Answer the question based on the context below. If you can't
    answer the question within the context provided, reply "I don't know". Do not try to answer outside the context provided.

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)

    # Initialize the LLM model and output parser
    model = Ollama(model="llama3.2")
    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain

def initPromptLLM():
    # Define the prompt template
    template = """
    Answer the question based on the context below. If you can't
    answer the question within the context provided, reply "I don't know". Do not try to answer outside the context provided.

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)

    # Initialize Ollama client-based model
    ollama_client = Ollama(model="llama3.2", base_url="http://34.44.184.36:11434")

    # Initialize output parser
    parser = StrOutputParser()

    # Create LLM chain
    chain = LLMChain(llm=ollama_client, prompt=prompt, output_parser=parser)

    return chain

# Function to retrieve and rerank documents based on the query
def retrieve_and_rerank(collection, query, top_k=5, rerank_top=3):
    login("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")
    # Load a pre-trained cross-encoder for ranking
    reranker = CrossEncoder("cross-encoder/msmarco-MiniLM-L6-en-de-v1")

    # Basic RAG: Retrieve top_k documents based on query embedding similarity
    query_embedding = embedding_model.encode(query).tolist()  # Convert to list
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_docs = results["documents"][0]
    
    if not retrieved_docs:
        return [("No relevant information found.", 0)]
    
    # Advanced RAG: Rerank the retrieved documents using a cross-encoder
    query_doc_pairs = [(query, doc) for doc in retrieved_docs]
    rerank_scores = reranker.predict(query_doc_pairs)
    
    # Normalize scores to be between 0 and 1
    min_score = min(rerank_scores)
    max_score = max(rerank_scores)
    if max_score - min_score > 0:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in rerank_scores]
    else:
        normalized_scores = [0] * len(rerank_scores)
    
    # Sort and select the top reranked results
    reranked_results = sorted(zip(retrieved_docs, normalized_scores), key=lambda x: x[1], reverse=True)
    
    return [(doc, score * 100) for doc, score in reranked_results[:rerank_top]]  # Convert scores to percentage

# Initialize embeddings and LLM chain only once and store in session state
if "collection" not in st.session_state:
    st.session_state.collection = create_embeddings()

# Initialize the prompt LLM chain
if "chain" not in st.session_state:
    st.session_state.chain = initPromptLLM()

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("AI Assistant")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:", key="user_input")
if user_input:
    # Retrieve and rerank documents based on the user query
    top_docs_with_scores = retrieve_and_rerank(st.session_state.collection, user_input)
    top_docs = [doc for doc, score in top_docs_with_scores]
    final_results = "\n".join(top_docs)
    
    # Generate response using the LLM chain
    response = st.session_state.chain.invoke({"context": final_results, "question": user_input})
    
    print("response before guard rail: ", response['text'])

    response = response['text']
    
    # GUARD RAIL IMPLEMENTATION - OUTPUT GUARD RAIL
    # Output guard rail to prevent hallucination or misleading information
    # Calculate the similarity score between the response and the context
    # If the response is "I don't know" or the similarity score is below 0.55, block the response
    response_embedding = embedding_model.encode(response).tolist()
    context_embedding = embedding_model.encode(final_results).tolist()
    similarity_score = torch.nn.functional.cosine_similarity(
        torch.tensor(response_embedding), torch.tensor(context_embedding), dim=0
    ).item()
    
    print("Similarity Score: ", similarity_score)
    if response == "I don't know" or similarity_score < 0.55:
        response = "I couldn't find relevant information based on the provided context."
        overall_confidence_score = 0  # Set confidence score to 0 if response is blocked by guard rail
    else:
        # Calculate the overall confidence score as the average of the top document scores
        if top_docs_with_scores:
            overall_confidence_score = sum(score for _, score in top_docs_with_scores) / len(top_docs_with_scores)
        else:
            overall_confidence_score = 0
    
    print("response after guard rail: ", response)
    
    # Append the user query, response, and confidence score to the chat history
    st.session_state.chat_history.append((user_input, response, overall_confidence_score))

# Display the chat history
for i, (user_q, ai_resp, confidence_score) in enumerate(st.session_state.chat_history):
    message(user_q, is_user=True, key=f"user_message_{i}")
    message(f"{ai_resp}\n\nConfidence Score: {confidence_score:.2f}%", is_user=False, key=f"ai_message_{i}")