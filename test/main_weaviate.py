import os
import fitz
import uvicorn
from typing import List
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.logger import logger

app = FastAPI()

# Constants
PDF_DIRECTORY = 'pdf_data'
# MODEL_PATH = "models/gemma-2-2b-it-Q6_K.gguf"
# MODEL_PATH = "models/llama-2-7b-chat.Q2_K.gguf"
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def load_pdfs_from_directory(directory: str) -> List[str]:
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                pdf_texts.append(text)
                logger.info(f"Loaded PDF: {filename}")
            else:
                logger.warning(f"Failed to load PDF: {filename}")
    logger.info(f"Total PDFs loaded: {len(pdf_texts)}")
    return pdf_texts

def initialize_rag_system():
  pdf_texts = load_pdfs_from_directory(PDF_DIRECTORY)
  if not pdf_texts:
    raise ValueError(f"No PDF texts were loaded. Check your '{PDF_DIRECTORY}' directory.")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
  all_splits = text_splitter.create_documents(pdf_texts)
  if not all_splits:
    raise ValueError("No text chunks were created. The PDFs might be empty.")

  # Connect to Weaviate instance
  client = weaviate.Client("http://<WEAVEATE_HOST>:<WEAVEATE_PORT>")  # Replace with your Weaviate instance details

  # Define the schema for your data (objects and their properties)
  schema = {
      "classes": [
          {
              "class": "Document",
              "properties": [
                  {"name": "content", "dataType": ["text"]},
              ]
          }
      ]
  }

  # Create the schema if it doesn't exist
  if not client.schema.exists("my_documents_schema"):
      client.schema.create("my_documents_schema", schema)

  # Weaviate doesn't directly accept embeddings, so we'll store the text itself in the "content" property
  for text_chunk in all_splits:
      client.data.create("Document", content=text_chunk)

  # Weaviate doesn't require a separate retriever component like Chroma
  retriever = None

  prompt_template = """Use the following pieces of context to answer the question at the end. 
                   Be concise and to the point. If you don't know the answer, just say "I don't know."

                   Context: {context}

                   Question: {question}

                   Answer: """

  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


  return RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      chain_type_kwargs={"prompt": prompt},
      retriever=retriever  # Set retriever to None
  )

# def initialize_rag_system():
#     pdf_texts = load_pdfs_from_directory(PDF_DIRECTORY)
#     if not pdf_texts:
#         raise ValueError(f"No PDF texts were loaded. Check your '{PDF_DIRECTORY}' directory.")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     all_splits = text_splitter.create_documents(pdf_texts)
#     if not all_splits:
#         raise ValueError("No text chunks were created. The PDFs might be empty.")

#     logger.info(f"Created {len(all_splits)} text chunks")

#     embeddings = HuggingFaceEmbeddings()
#     db = Chroma.from_documents(all_splits, embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY)
#     db.persist()

#     logger.info("Created and persisted Chroma database")

#     llm = LlamaCpp(
#         model_path=MODEL_PATH,
#         temperature=0.1,
#         max_tokens=256,
#         n_ctx=2048,
#         top_p=0.95,
#         verbose=True,
#         n_threads=8,  # Adjust based on your CPU
#     )

#     retriever = db.as_retriever(search_kwargs={"k": 3})

#     prompt_template = """Use the following pieces of context to answer the question at the end. 
#                          Be concise and to the point. If you don't know the answer, just say "I don't know."

#                         Context: {context}

#                         Question: {question}

#                         Answer: """

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt}
#     )

qa_chain = initialize_rag_system()

def safe_retrieve(query: str, retriever):
    docs = retriever.get_relevant_documents(query)
    if not docs:
        logger.warning(f"No relevant documents found for query: {query}")
        return "No relevant documents found for the query."
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    for i, doc in enumerate(docs):
        logger.info(f"Document {i+1} content: {doc.page_content[:100]}...")  # Log first 100 chars of each doc
    return docs

def query_system(query: str):
    try:
        logger.info(f"Processing query: {query}")
        context = safe_retrieve(query, qa_chain.retriever)
        if isinstance(context, str):
            return context 
        logger.info("Invoking LLM...")
        result = qa_chain.invoke(query)
        logger.info(f"Raw LLM output: {result}")
        response = result.get('result', '')
        if not response.strip():
            logger.warning("LLM generated an empty response")
            return "The system couldn't generate a response. Please try rephrasing your question."
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}")
        return f"An error occurred: {str(e)}"

class Query(BaseModel):
    text: str

@app.get("/")
async def root():
    logger.info("Server is up and running")
    return {"message": "Server is up and running"}

@app.post("/query")
async def query_endpoint(query: Query):
    try:
        logger.info(f"User query: {query.text}")
        result = query_system(query.text)
        logger.info(f"System response: {result}")
        return {"response": result}
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting the server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)