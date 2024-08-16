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
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.logger import logger

app = FastAPI()

# Constants
FILE_DIRECTORY = 'data'
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

def load_text_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode the file {file_path} with any of the attempted encodings")

class CustomTextLoader(TextLoader):
    def __init__(self, file_path: str, encoding: str = None):
        super().__init__(file_path, encoding)

    def load(self) -> List[Document]:
        try:
            text = load_text_with_fallback_encoding(self.file_path)
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Error loading {self.file_path}: {str(e)}")
            return []

    def lazy_load(self):
        try:
            text = load_text_with_fallback_encoding(self.file_path)
            metadata = {"source": self.file_path}
            yield Document(page_content=text, metadata=metadata)
        except Exception as e:
            logger.error(f"Error lazy loading {self.file_path}: {str(e)}")

def initialize_rag_system():
    loader = DirectoryLoader(FILE_DIRECTORY, glob="./*.txt", loader_cls=CustomTextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents were loaded. Check your 'new_articles' directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)

    if not all_splits:
        raise ValueError("No text chunks were created. The documents might be empty.")

    logger.info(f"Created {len(all_splits)} text chunks")

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY)
    db.persist()

    logger.info("Created and persisted Chroma database")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_tokens=256,
        n_ctx=2048,
        top_p=0.95,
        verbose=True,
        n_threads=8,  # Adjust based on your CPU
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Use the following pieces of context to answer the question at the end. 
                         Be concise and to the point. If you don't know the answer, just say "I don't know."

                        Context: {context}

                        Question: {question}

                        Answer: """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Rest of the code remains the same...

# Defining the query-retrieval chain
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