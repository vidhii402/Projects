from fastapi import FastAPI, HTTPException
from research_agent import ResearchRequest, ResearchResponse, process_research_request
import uvicorn
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Assistant",
    description="A research and summarization tool using Claude",
    version="1.0.0"
)

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Process a research request and return findings."""
    try:
        logger.info(f"Received research request: {request.query}")
        response = await process_research_request(request)
        logger.info("Request processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 