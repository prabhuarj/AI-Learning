import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.groq import Groq

# Node parsing / preprocessing
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser

#Settings
from llama_index.core import Settings

# Retrievers
from llama_index.core.retrievers import VectorIndexRetriever

# Response synthesizer + modes
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

# Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

FOLDER_PATH = r"C:\AI Project\userguide"
STORAGE_DIR = r"C:\AI Project\storage"

# Initialize Groq LLM model
llm = Groq(model="llama-3.3-70b-versatile", temperature=0.0)
print("LLM initialized:", llm)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model initialized:", embed_model)

# Node parser: splits documents into nodes/chunks
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=150)

# Set defaults globally
Settings.llm = llm
Settings.node_parser = node_parser
Settings.embed_model = embed_model

# Load all text files from docs/ folder
documents = SimpleDirectoryReader(FOLDER_PATH).load_data()
print(f"Loaded {len(documents)} documents.")
print("Example document text:\n", documents[1].text[:300])

index = VectorStoreIndex.from_documents(documents)

# Persist (save) the index to disk
storage_dir = STORAGE_DIR
index.storage_context.persist(persist_dir=storage_dir)
print(f"Index persisted to directory: {storage_dir}")


# Reload index from storage
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
index_reloaded = load_index_from_storage(storage_context)
print("Index reloaded.")

# Build retriever using the vector index
retriever = VectorIndexRetriever(index=index_reloaded, similarity_top_k=4)
print("Retriever created:")

# Create a response synthesizer: choose a mode
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,  # alternatives: REFINE, COMPACT, TREE_SUMMARIZE, etc.
    streaming=False,  
)

# query_engine = RetrieverQueryEngine.from_args(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     streaming=False,
# )


qa_template = PromptTemplate(
"""
You are a highly STRICT Retrieval-Augmented Generation (RAG) assistant.

### RULES (Follow these rules EXACTLY):
1. You MUST use ONLY the information from the provided CONTEXT.
2. The ONLY acceptable response when the answer is NOT found in CONTEXT is the exact phrase: "I don't know".
3. ABSOLUTELY DO NOT use external or pre-trained knowledge. Base your answer PURELY on the CONTEXT.
4. Do NOT assume, guess, infer, or create information not explicitly stated in the CONTEXT.
5. Do NOT expand or elaborate beyond the CONTEXT.
6. If the question is partially answered by context, answer ONLY the part supported by context.

### CONTEXT:
{context_str}

### QUESTION:
{query_str}

### STRICT ANSWER:
"""
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    text_qa_template=qa_template,
    streaming=False,
)

def rag_tool(query: str) -> str:
    response = query_engine.query(query)
    return str(response)