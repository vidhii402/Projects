import os
from typing import Dict, List, Tuple, Any, Optional, TypedDict, Annotated
from dotenv import load_dotenv
import operator
from functools import partial

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, ToolInvocation

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State for the agent graph."""
    query: str
    context: List[str]
    chat_history: List[Tuple[str, str]]
    search_result: Optional[List[str]]
    summary: Optional[str]
    answer: Optional[str]
    thoughts: Optional[str]
    follow_up: Optional[str]
    analysis_needed: Optional[bool]
    tools_to_use: Optional[List[str]]
    current_tool: Optional[str]
    tool_input: Optional[Dict]
    tool_output: Optional[str]
    final_response: Optional[str]

def initialize_llm():
    """Initialize the Claude LLM."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Please add it to your .env file.")
    
    return ChatAnthropic(
        api_key=api_key,
        model="claude-3-sonnet-20240229",
        temperature=0.2,
        max_tokens=1000
    )

def initialize_retriever():
    """Initialize the document retriever with enhanced context compression."""
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if vector DB exists
    if not os.path.exists("data/chroma_db"):
        raise FileNotFoundError(
            "Vector database not found. Please run ingest.py first to process PDFs."
        )
    
    # Load vector DB
    vectordb = Chroma(
        collection_name="pdf_assistant",
        embedding_function=hf_embeddings,
        persist_directory="data/chroma_db"
    )
    
    # Create a basic retriever
    basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # Initialize LLM for document compression
    llm = initialize_llm()
    
    # Create document compressors
    llm_extractor = LLMChainExtractor.from_llm(llm)
    embeddings_filter = EmbeddingsFilter(
        embeddings=hf_embeddings,
        similarity_threshold=0.7
    )
    
    # Create a pipeline of document compressors
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[embeddings_filter, llm_extractor]
    )
    
    # Create enhanced retriever with compression
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=basic_retriever
    )

def create_search_tool(retriever):
    """Create a tool for searching documents."""
    return Tool(
        name="SearchDocuments",
        description="Search for relevant information in the PDF documents. Input should be a specific question or keywords.",
        func=lambda query: [doc.page_content for doc in retriever.get_relevant_documents(query)]
    )

def create_summarize_tool(llm):
    """Create a tool for summarizing document content."""
    summarize_prompt = PromptTemplate.from_template(
        "Summarize the following content in a concise way:\n\n{content}\n\nSummary:"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    
    return Tool(
        name="SummarizeContent",
        description="Summarize a piece of content to make it more digestible. Input should be the content to summarize.",
        func=lambda content: summarize_chain.invoke({"content": content})
    )

def create_analyze_tool(llm):
    """Create a tool for analyzing and extracting specific information."""
    analyze_prompt = PromptTemplate.from_template(
        "Analyze the following content and extract specific information about {question}:\n\n{content}\n\nAnalysis:"
    )
    
    def analyze_content(inputs):
        formatted_prompt = analyze_prompt.format(
            question=inputs["question"],
            content=inputs["content"]
        )
        return llm.invoke(formatted_prompt).content
    
    return Tool(
        name="AnalyzeContent",
        description="Analyze content to extract specific information or insights. Input should be a JSON with 'question' and 'content' keys.",
        func=analyze_content
    )

def plan_approach(state: AgentState) -> AgentState:
    """Plan the approach to answering the query."""
    llm = initialize_llm()
    
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a planning agent for a PDF research assistant. 
        Your job is to understand the user's query and decide what tools to use and in what order.
        Available tools:
        - SearchDocuments: Search for relevant information in the PDFs
        - SummarizeContent: Summarize content to make it digestible
        - AnalyzeContent: Extract specific information or insights
        
        Think step by step about how to approach the query.
        Consider:
        1. Do we need to search for information first?
        2. Should we summarize the results?
        3. Do we need deep analysis of specific aspects?
        
        Choose the most effective approach considering the context and chat history."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Query: {query}\n\nWhat's the best approach to answer this?")
    ])
    
    chat_history_messages = []
    for human, ai in state.get("chat_history", []):
        chat_history_messages.append(HumanMessage(content=human))
        chat_history_messages.append(AIMessage(content=ai))
    
    plan_chain = plan_prompt | llm | StrOutputParser()
    
    thoughts = plan_chain.invoke({
        "query": state["query"],
        "chat_history": chat_history_messages
    })
    
    # Determine if search is needed
    search_needed = "search" in thoughts.lower() or "find" in thoughts.lower()
    
    # Determine if analysis is needed
    analysis_needed = "analyze" in thoughts.lower() or "extract" in thoughts.lower()
    
    # Determine tools to use
    tools_to_use = []
    if search_needed:
        tools_to_use.append("SearchDocuments")
    if "summarize" in thoughts.lower():
        tools_to_use.append("SummarizeContent")
    if analysis_needed:
        tools_to_use.append("AnalyzeContent")
    
    # Update state
    state["thoughts"] = thoughts
    state["analysis_needed"] = analysis_needed
    state["tools_to_use"] = tools_to_use
    
    # Set the first tool to use
    if tools_to_use:
        state["current_tool"] = tools_to_use[0]
    
    return state

def route_to_tool(state: AgentState) -> str:
    """Route to the appropriate tool or to the final answer."""
    if not state.get("tools_to_use"):
        return "generate_answer"
    
    current_tool = state.get("current_tool")
    if current_tool == "SearchDocuments":
        return "search_documents"
    elif current_tool == "SummarizeContent":
        return "summarize_content"
    elif current_tool == "AnalyzeContent":
        return "analyze_content"
    else:
        return "generate_answer"

def search_documents(state: AgentState) -> AgentState:
    """Search documents for relevant information."""
    retriever = initialize_retriever()
    search_tool = create_search_tool(retriever)
    
    # Execute search
    results = search_tool.func(state["query"])
    
    # Update state
    state["search_result"] = results
    
    # Update tools - remove current and move to next
    tools = state.get("tools_to_use", [])
    if tools and tools[0] == "SearchDocuments" and len(tools) > 1:
        state["current_tool"] = tools[1]
    else:
        state["tools_to_use"] = []
    
    return state

def summarize_content(state: AgentState) -> AgentState:
    """Summarize the retrieved content."""
    llm = initialize_llm()
    summarize_tool = create_summarize_tool(llm)
    
    # Get content to summarize
    content = "\n\n".join(state.get("search_result", []))
    
    # Execute summarization
    summary = summarize_tool.func(content)
    
    # Update state
    state["summary"] = summary
    
    # Update tools - remove current and move to next
    tools = state.get("tools_to_use", [])
    if tools and tools[0] == "SummarizeContent" and len(tools) > 1:
        state["current_tool"] = tools[1]
    else:
        state["tools_to_use"] = []
    
    return state

def analyze_content(state: AgentState) -> AgentState:
    """Analyze the content for specific information."""
    llm = initialize_llm()
    analyze_tool = create_analyze_tool(llm)
    
    # Prepare input for analysis
    content_to_analyze = state.get("summary", "") or "\n\n".join(state.get("search_result", []))
    
    analysis_input = {
        "question": state["query"],
        "content": content_to_analyze
    }
    
    # Execute analysis
    analysis = analyze_tool.func(analysis_input)
    
    # Update state
    state["tool_output"] = analysis
    
    # Update tools - remove current and move to next
    tools = state.get("tools_to_use", [])
    if tools and tools[0] == "AnalyzeContent" and len(tools) > 1:
        state["current_tool"] = tools[1]
    else:
        state["tools_to_use"] = []
    
    return state

def generate_answer(state: AgentState) -> AgentState:
    """Generate a comprehensive answer based on all gathered information."""
    llm = initialize_llm()
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful PDF research assistant. Create a comprehensive answer to the user's query
        using all the information gathered. Be specific, accurate and cite information from the documents when possible.
        If you're not sure about something, be transparent about limitations. Your answer should be well-structured,
        direct, and exactly address the user's query."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Query: {query}
        
        Available information:
        - Search Results: {search_results}
        - Summary: {summary}
        - Analysis: {analysis}
        
        Please provide a comprehensive answer:""")
    ])
    
    chat_history_messages = []
    for human, ai in state.get("chat_history", []):
        chat_history_messages.append(HumanMessage(content=human))
        chat_history_messages.append(AIMessage(content=ai))
    
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    # Gather all available information
    answer = answer_chain.invoke({
        "query": state["query"],
        "search_results": "\n\n".join(state.get("search_result", [])) if state.get("search_result") else "No search results available.",
        "summary": state.get("summary", "No summary available."),
        "analysis": state.get("tool_output", "No analysis available."),
        "chat_history": chat_history_messages
    })
    
    # Update state
    state["answer"] = answer
    
    # Generate follow-up suggestions
    follow_up_prompt = PromptTemplate.from_template(
        "Based on the user's query: '{query}' and our answer, suggest 3 follow-up questions they might ask next:"
    )
    follow_up_chain = follow_up_prompt | llm | StrOutputParser()
    
    follow_up = follow_up_chain.invoke({"query": state["query"]})
    state["follow_up"] = follow_up
    
    # Construct final response
    final_response = f"{answer}\n\n"
    if state.get("follow_up"):
        final_response += f"\n\nPossible follow-up questions:\n{follow_up}"
    
    state["final_response"] = final_response
    
    return state

def create_pdf_agent():
    """Create the PDF agent workflow."""
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_approach)
    workflow.add_node("search_documents", search_documents)
    workflow.add_node("summarize_content", summarize_content)
    workflow.add_node("analyze_content", analyze_content)
    workflow.add_node("generate_answer", generate_answer)
    
    # Add edges
    workflow.add_edge("plan", route_to_tool)
    workflow.add_edge("search_documents", route_to_tool)
    workflow.add_edge("summarize_content", route_to_tool)
    workflow.add_edge("analyze_content", route_to_tool)
    workflow.add_edge("generate_answer", END)
    
    # Set entrypoint
    workflow.set_entry_point("plan")
    
    # Compile the graph
    return workflow.compile()

def process_query(query: str, chat_history: List[Tuple[str, str]] = None):
    """Process a user query using the PDF agent."""
    if chat_history is None:
        chat_history = []
    
    # Initialize state
    initial_state = AgentState(
        query=query,
        context=[],
        chat_history=chat_history,
        search_result=None,
        summary=None,
        answer=None,
        thoughts=None,
        follow_up=None,
        analysis_needed=None,
        tools_to_use=None,
        current_tool=None,
        tool_input=None,
        tool_output=None,
        final_response=None
    )
    
    # Create and run the agent
    agent = create_pdf_agent()
    result = agent.invoke(initial_state)
    
    # Return final answer and state
    return result.get("final_response"), result

def main():
    """Main function to run the PDF agent interactively."""
    try:
        print("Initializing PDF Agent...")
        
        # Check if required resources exist
        if not os.path.exists("data/chroma_db"):
            print("Vector database not found. Please run ingest.py first to process PDFs.")
            return
            
        # Set environment variable TOKENIZERS_PARALLELISM to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("PDF Agent ready! Ask me anything about the ingested PDFs.")
        print("Type 'exit' to quit.")
        
        chat_history = []
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("\nThinking...")
            try:
                answer, state = process_query(query, chat_history)
                print(f"\nAgent:\n{answer}")
                
                # Update chat history
                chat_history.append((query, state.get("answer", "")))
                
                # Keep only last 5 exchanges to prevent context growth
                if len(chat_history) > 5:
                    chat_history = chat_history[-5:]
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try a different question.")
                
    except Exception as e:
        print(f"Error initializing agent: {e}")

if __name__ == "__main__":
    main()