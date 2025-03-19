# Agentic PDF Research Assistant

An intelligent, agentic system that transforms PDFs into an interactive knowledge base you can query in natural language.

## Features

- **Agentic Architecture**: Dynamic planning and tool selection based on query needs
- **Semantic Understanding**: Goes beyond keyword search to comprehend document meaning
- **Multi-stage Reasoning**: Plans, searches, analyzes, and synthesizes information automatically
- **Context Awareness**: Maintains conversation history for natural follow-ups
- **Self-reflection**: Generates thoughts about approach and includes reasoning in responses
- **Follow-up Suggestions**: Automatically suggests relevant follow-up questions

## How It Works

This system implements a true agentic workflow that:

1. **Plans the approach** to answering each query
2. **Selects appropriate tools** based on query complexity
3. **Searches and retrieves** relevant document sections
4. **Summarizes long content** when beneficial
5. **Analyzes information** to extract specific insights
6. **Synthesizes a comprehensive answer** from all gathered information
7. **Suggests follow-up questions** to deepen the conversation

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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

2. Then run the agentic PDF assistant:
   ```bash
   python pdf_agent.py
   ```

3. Ask questions about the contents of the PDFs. Type 'exit' to quit.

## Project Structure

- `ingest.py` - Processes PDFs and creates the vector database
- `pdf_agent.py` - Agentic workflow for PDF question answering
- `query.py` - Simple interactive CLI for asking questions (non-agentic version)
- `docs/` - Place to store PDF documents
- `data/chroma_db/` - Vector database storage (created automatically)

## Technical Architecture

The system uses a LangGraph-based workflow with:

- **State Management**: Tracks the entire reasoning process
- **Dynamic Routing**: Selects tools based on query needs
- **Tool Integration**: SearchDocuments, SummarizeContent, and AnalyzeContent tools
- **Enhanced Retrieval**: ContextualCompressionRetriever with document filtering
- **Planning Stage**: Deliberate reasoning about approach before execution

## Requirements

- Python 3.8+
- API key from Anthropic (https://console.anthropic.com/)
- PDFs to analyze
- langchain
- langgraph
- langchain-anthropic
- langchain-chroma
- langchain-huggingface