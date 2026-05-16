from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

PDF_PATH = os.getenv("PDF_PATH")
file_path = "./document.pdf";
loader = PyPDFLoader(file_path)
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="documentos",
    connection="postgresql+psycopg://postgres:postgres@localhost:5432/rag",
    use_jsonb=True
)

def ingest_pdf():    
    docs = loader.load()
    all_splits = text_spliter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)

if __name__ == "__main__":
    ingest_pdf()