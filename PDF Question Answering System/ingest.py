from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os, glob

load_dotenv()

def main():
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        print("No PDF files found in docs/ folder.")
        return

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        all_documents.extend(chunks)

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        collection_name="pdf_assistant",
        embedding_function=hf_embeddings,
        persist_directory="data/chroma_db"
    )

    vectordb.add_documents(all_documents)
    vectordb.persist()  

    print("\nIngestion complete.\n")

if __name__ == "__main__":
    main()