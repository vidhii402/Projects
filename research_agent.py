from typing import Dict, List, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import logging
from anthropic import Anthropic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logger.info(f"API Key present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
logger.info(f"API Key starts with: {os.getenv('ANTHROPIC_API_KEY')[:15]}...")

def analyze_with_claude(query: str, results: str) -> str:
    """Analyze results using Claude."""
    try:
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        logger.info("Attempting to use claude-3-haiku model")
        message = client.messages.create(
            model="claude-3-haiku",  # Using the model you have access to
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Please analyze these search results about '{query}' and provide:
                1. Brief Summary (2-3 sentences)
                2. Key Points (3-4 bullet points)
                3. Detailed Analysis
                4. Conclusion

                Search Results: {results}"""
            }]
        )
        logger.info("Successfully got response from Claude")
        return message.content
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        return f"Claude analysis unavailable. Raw results:\n{results}"

class ResearchRequest(BaseModel):
    """Input model for research requests."""
    query: str = Field(..., description="The research query or topic")

class ResearchResponse(BaseModel):
    """Output model for research results."""
    summary: str = Field(..., description="Final research summary")
    error: str = Field(default="", description="Error message if any")

async def process_research_request(request: ResearchRequest) -> ResearchResponse:
    """Process a research request and return results."""
    try:
        logger.info(f"Processing research request: {request.query}")
        
        # Initialize search tool
        search = DuckDuckGoSearchRun()
        
        # Execute search
        logger.info("Executing search...")
        search_results = search.run(request.query)
        logger.info("Search completed")

        # Try to analyze with Claude
        logger.info("Analyzing with Claude...")
        analysis = analyze_with_claude(request.query, search_results)
        logger.info("Analysis completed")
        
        return ResearchResponse(
            summary=analysis,
            error=""
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return ResearchResponse(
            summary="",
            error=f"Error processing research request: {str(e)}"
        ) 