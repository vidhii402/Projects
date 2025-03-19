# PDF Question Answering System

A system that ingests PDF documents and allows you to ask questions about their contents using LangChain and Claude.

## Features

- PDF document loading and processing
- Vector embeddings using Hugging Face models
- Semantic search with Chroma vector database
- Interactive Q&A using Claude AI

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Create a `docs` folder and add PDF files you want to analyze:
   ```bash
   mkdir docs
   # Add your PDF files to this folder
   ```

## Usage

1. First, ingest the PDF documents:
   ```bash
   python ingest.py
   ```

2. Then, start the interactive query system:
   ```bash
   python query.py
   ```

3. Ask questions about the contents of the PDFs. Type 'exit' to quit.

## Project Structure

- `ingest.py` - Processes PDFs and creates the vector database
- `query.py` - Interactive CLI for asking questions about the PDFs
- `docs/` - Place to store PDF documents
- `data/chroma_db/` - Vector database storage (created automatically)
- `langgraph_flow.yaml` - LangGraph workflow definition

## Requirements

- Python 3.8+
- API key from Anthropic (https://console.anthropic.com/)
- PDFs to analyze

## Note

Make sure your Anthropic API key is valid and has proper permissions. If you're using the API for the first time, you may need to add payment information to your Anthropic account.