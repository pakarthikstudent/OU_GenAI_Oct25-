from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
import oracledb


# --------------------------------------------------
# 1 SETUP FUNCTION — run once to create the vector store
# --------------------------------------------------
def setup_oracle_vectorstore():
    # Load and split documents
    loader = TextLoader("my.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # Embedding model 
    embeddings = OllamaEmbeddings(model="gemma2:2b")

    # Oracle client pool
    client = oracledb.create_pool(
        user="student",
        password="apelix",
        dsn="localhost:1521/FREEPDB1",
        min=1,
        max=5,
        increment=1,
    )

    # Create vector store
    vector_store = OracleVS.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        table_name="LANGCHAIN_VSTORE",
        distance_strategy=DistanceStrategy.COSINE,
    )

    print("Oracle Vector Store setup complete.")
    return vector_store


# --------------------------------------------------
# 2️ SEARCH FUNCTION — call this repeatedly for queries
# --------------------------------------------------
def search_oracle_vector(vector_store, query, k=3, threshold=0.3):
    
    results = vector_store.similarity_search_with_score(query, k=k)
    # Filter by threshold score
    filtered = [r for r, score in results if score >= threshold]

    if not filtered:
        print(f" No relevant match found for query: '{query}'")
    else:
        print(f" Found {len(filtered)} relevant result(s):\n")
        for r in filtered:
            print(r.page_content[:200], "\n---\n")


# --------------------------------------------------
# 3️ MAIN — Example usage
# --------------------------------------------------
if __name__ == "__main__":
    vs = setup_oracle_vectorstore()

    # Example queries
    search_oracle_vector(vs, "what is langchain?")
    search_oracle_vector(vs, "45?")
