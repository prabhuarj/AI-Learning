import os
import shutil
import json
from typing import List, Dict, Union
from dotenv import load_dotenv
load_dotenv()


# ====================================
# 1️⃣ Document Loading and Preprocessing
# ====================================
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

FOLDER_PATH = r"C:\AI Project\userguide"
CHROMA_DB_DIR = r"C:\AI Project\chroma_db"

print("Step 1: Loading documents...")
loader = DirectoryLoader(FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

print("Step 2: Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)
print(f"Total number of documents: {len(documents)}")
print(f"Created {len(split_documents)} text chunks.")
print("\nDocument splitting complete.")
print("Example document text:\n", split_documents[1].page_content[:300])


if os.path.exists(CHROMA_DB_DIR):
    print(f"Clearing existing Chroma DB at {CHROMA_DB_DIR}...")
    shutil.rmtree(CHROMA_DB_DIR)

print("Step 3: Creating embeddings and vector store...")
embeddings_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(split_documents, embedding=embeddings_model, persist_directory=CHROMA_DB_DIR)
print("\nIngestion complete. The vector store is ready for retrieval.")
retriever = vector_store.as_retriever()
print("Vector store ready for retrieval.")

# ====================================
# 2️⃣ LLM and Prompt Definition
# ====================================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)


prompt_template = ChatPromptTemplate.from_template("""

You are an intelligent assistant.
Use ONLY the provided context to answer the question.


# You are a Cloud Access Guide Assistant that helps users with access, provisioning, and connectivity questions for services like AWS EC2, Azure VM, GCP VM, vSphere etc.

# Use ONLY the information from the retrieved documents to answer. 
# If the answer is not found in the documents, reply:
# "I don’t have information about that in the access guide."

# FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

# Answer:
# <clear step-by-step instructions or explanation based on context>

# If Steps Apply:
# 1. Step one...
# 2. Step two...
# 3. Step three...

# Notes:
# - Any warnings or limitations from guide

# Sources:
# - <file_name> (optional)

# Citations:
# [source:<file_path>::<chunk_index>] Short note about what was used from that source.
# [source:filename.pdf::page]

# Do NOT include JSON, metadata, developer notes, hallucinated values, or any text not grounded in the documents.
""")



# ====================================
# 4️⃣ Build RAG Chain
# ====================================
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_template
    | llm
)

# ====================================
# 5️⃣ Define RAG Tool Function
# ====================================
def rag_tool(question: str):
    """Fetch relevant context and generate an answer using the RAG pipeline."""
    print(f"\n Running RAG for question: {question}")
    result = rag_chain.invoke(question)
    # return json.dumps(result.dict(), indent=2)
    return result
