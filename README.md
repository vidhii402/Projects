# AI Research Assistant

An AI-powered research and summarization tool built with LangChain, LangGraph, and GPT-4. This tool demonstrates agentic workflows by combining retrieval, reasoning, and report generation capabilities.

## Features

- Automated research on any topic using GPT-4
- Intelligent search and information retrieval
- Advanced analysis and synthesis of findings
- Structured workflow using LangGraph
- FastAPI-based REST API interface

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Make a research request:
```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What are the latest developments in quantum computing?",
           "openai_api_key": "your_api_key_here"
         }'
```

## Architecture

The application uses a structured workflow powered by LangGraph:

1. **Search and Analysis**: Uses Tavily search tool and GPT-4 to gather and analyze information
2. **Synthesis**: Processes and synthesizes the findings into a comprehensive summary
3. **Output Generation**: Produces a structured research report

## Advanced Features

- Error handling and recovery
- Structured workflow management
- API-first design
- Extensible architecture

## Contributing

Feel free to submit issues and enhancement requests! 