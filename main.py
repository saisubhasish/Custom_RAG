import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from src.logger import logger
from src.utils import query_system, initialize_rag_system

app = FastAPI()

# Constants
FILE_DIRECTORY = 'data'
# MODEL_PATH = "models/gemma-2-2b-it-Q6_K.gguf"
# MODEL_PATH = "models/llama-2-7b-chat.Q2_K.gguf"
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"


class Query(BaseModel):
    text: str

@app.get("/")
async def root():
    logger.info("Server is up and running")
    return {"message": "Server is up and running"}

@app.post("/query")
async def query_endpoint(query: Query):
    try:
        qa_chain = initialize_rag_system(MODEL_PATH, FILE_DIRECTORY, CHROMA_PERSIST_DIRECTORY)
        logger.info(f"QA chain created: {qa_chain}")
        logger.info(f"User query: {query.text}")
        result = query_system(qa_chain, query.text)
        logger.info(f"System response: {result}")
        
        # Get the retrieved documents for debugging
        docs = qa_chain.retriever.get_relevant_documents(query.text)
        context = "\n".join([doc.page_content for doc in docs])
        
        return {
            "response": result,
            "debug_info": {
                "retrieved_context": context
            }
        }
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)