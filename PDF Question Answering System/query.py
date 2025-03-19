from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
import os
import sys

load_dotenv()

def main():
    """
    1. Load the local Chroma DB with stored embeddings.
    2. Build a ConversationalRetrievalChain using Claude for Q&A.
    3. Let the user type queries in a loop.
    4. Return summarized or direct Q&A from the PDF content.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Environment variable ANTHROPIC_API_KEY not found.")
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir())
        if os.path.exists(".env"):
            print(".env file exists.")
            
        api_key = input("Please enter your Anthropic API key: ")
        with open(".env", "w") as f:
            f.write(f"ANTHROPIC_API_KEY={api_key}")
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    print("Using API key:", api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if api_key else "None")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        print("Testing API key validity...")
        test_llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            max_tokens=10
        )
        test_llm.invoke("Say OK")
        print("API key validated successfully!")
    except Exception as e:
        print(f"Error validating API key: {e}")
        print("\nYour API key appears to be invalid. Please get a valid API key from https://console.anthropic.com/")
        print("Exiting program.")
        sys.exit(1)

    print("\nLoading embeddings and vector database...")
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists("data/chroma_db"):
        print("\nERROR: Vector database not found at data/chroma_db")
        print("Have you run the ingest.py script first to process your PDFs?")
        print("Please run 'python ingest.py' before using this script.")
        sys.exit(1)

    # Loading vector DB
    try:
        vectordb = Chroma(
            collection_name="pdf_assistant",
            embedding_function=hf_embeddings,
            persist_directory="data/chroma_db"
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("\nMake sure you've run ingest.py to create the database.")
        sys.exit(1)

    print("Setting up Claude for question answering...")
    # Using Claude via ChatAnthropic
    llm = ChatAnthropic(
        api_key=api_key,
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.2
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    chat_history = []
    print("\nPDF Assistant (Claude): Ask me anything about the ingested PDFs!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            result = qa_chain.invoke({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            chat_history.append((query, answer))
            print(f"\nAssistant:\n{answer}\n")
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try a different question.")

if __name__ == "__main__":
    main()
