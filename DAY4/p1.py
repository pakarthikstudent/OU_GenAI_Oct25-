# Step 0 - Imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
import oracledb

# Step 1 - Load your text file
loader = TextLoader("my_docs.txt")
documents = loader.load()

# Step 2 - Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3 - Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="gemma2:2b")

# Step 4 - Connect to Oracle 23ai
connection = oracledb.connect(
    user="student",
    password="apelix",
    dsn="localhost:1521/FREEPDB1"
)

#  Step 5 - Use the connection itself as the client
client = connection

# Step 6 - Create or use an existing Oracle Vector Store
vector_store = OracleVS.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,                               # ✅ pass connection, not cursor
    table_name="LANGCHAIN_VSTORE",
    distance_strategy=DistanceStrategy.COSINE     # ✅ required
)

# Step 7 - Query your data using vector similarity search
query = "what is langchain?"
results = vector_store.similarity_search(query, k=3)

print(f"Found {len(results)} relevant documents:\n")
for r in results:
    print(r.page_content[:200], "\n---\n")

# Step 8 - Close connection
connection.close()
