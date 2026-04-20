from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# function to load PDF documents
DATA_PATH = "data/"
def load_pdf_documents(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents
documents = load_pdf_documents(data=DATA_PATH)

# function to split documents into chunks
def create_chunks(extracted_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                   chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_documents)
    return chunks
text_chunks = create_chunks(extracted_documents=documents)
#print(f"Number of chunks created: {len(text_chunks)}")

# function to create embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embadedding_model = get_embedding_model()
# create vector store
DB_PATH = "vector_store/faiss_index"
db=FAISS.from_documents(text_chunks, embadedding_model)
db.save_local(DB_PATH)
