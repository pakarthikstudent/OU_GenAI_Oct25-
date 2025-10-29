import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import ChainInput, AuthContext
from langchain_groq import ChatGroq

# --- Step 1: Load and split documents ---
loader = TextLoader("my_docs.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = splitter.split_documents(documents)

# Optionally, add Pebblo metadata
for d in docs:
    d.metadata = {
        "authorized_identities": ["user", "A"],
        "pebblo_semantic_topics": ["AI", "framework"],
        "pebblo_semantic_entities": ["LangChain", "Groq"]
    }

# --- Step 2: Build FAISS index ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# --- Step 3: Initialize LLM ---
llm = ChatGroq(model="groq/compound-mini", api_key=os.getenv("GROQ_API_KEY"))

# --- Step 4: Build Pebblo RetrievalQA ---
qa_chain = PebbloRetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    app_name="codespace-pebblo-demo",
    description="Local Pebblo + FAISS RAG Example",
    owner="codespace-user",
    chain_type="stuff",
    verbose=True
)

# --- Step 5: Ask question as 'alice' ---
query_input = ChainInput(
    query="What is LangChain?",
    auth_context=AuthContext(user_id="user")
)

result = qa_chain.invoke(query_input.dict())
print("\nAnswer:", result)
