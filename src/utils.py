import os
import fitz
import pandas as pd
from typing import List
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader

from logger import logger

def initialize_rag_system(MODEL_PATH, FILE_DIRECTORY, CHROMA_PERSIST_DIRECTORY):
    # For any one file type (eg. pdf, text, csv) use the code below
    # Otherwise delete the 'chroma_db' directory and and rerun it in case you are changing the file type

    # Check if Chroma database already exists 
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        logger.info("Loading existing Chroma database")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        # # Loading pdf files
        # all_splits = load_pdf_files(FILE_DIRECTORY)
        
        # # Loading text files
        # all_splits = load_text_files(FILE_DIRECTORY)

        # Loading CSV files
        all_splits = load_csv_files(FILE_DIRECTORY)

        logger.info(f"Created {len(all_splits)} text chunks")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY)
        db.persist()

    logger.info("Chroma database ready")

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

def load_pdf_files(FILE_DIRECTORY):
    pdf_texts = load_pdfs_from_directory(FILE_DIRECTORY)

    if not pdf_texts:
        raise ValueError(f"No PDF texts were loaded. Check your '{FILE_DIRECTORY}' directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_splits = text_splitter.create_documents(pdf_texts)

    if not all_splits:
        raise ValueError("No text chunks were created. The PDFs might be empty.")
    return all_splits

def load_csv_files(FILE_DIRECTORY):
    csv_documents = []
    csv_files_found = False
    
    logger.info(f"Searching for CSV files in directory: {FILE_DIRECTORY}")
    
    for filename in os.listdir(FILE_DIRECTORY):
        if filename.endswith('.csv'):
            csv_files_found = True
            csv_path = os.path.join(FILE_DIRECTORY, filename)
            logger.info(f"Found CSV file: {csv_path}")
            
            try:
                # First, try to read the CSV file using pandas
                df = pd.read_csv(csv_path)
                logger.info(f"Successfully read {filename} with pandas. Shape: {df.shape}")
                
                try:
                    # Try loading with CSVLoader
                    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
                    csv_documents.extend(loader.load())
                    logger.info(f"Loaded CSV with CSVLoader: {filename}")
                except Exception as csv_loader_error:
                    logger.error(f"Error loading {filename} with CSVLoader: {str(csv_loader_error)}")
                    logger.info(f"Falling back to pandas for {filename}")
                    
                    # Fallback to using pandas if CSVLoader fails
                    for _, row in df.iterrows():
                        content = ", ".join(f"{col}: {val}" for col, val in row.items())
                        csv_documents.append(Document(page_content=content, metadata={"source": filename}))
                    logger.info(f"Loaded CSV with pandas fallback: {filename}")
                
            except Exception as e:
                logger.error(f"Unexpected error loading {filename}: {str(e)}")
    
    if not csv_files_found:
        logger.warning(f"No CSV files found in {FILE_DIRECTORY}")
        return []
    
    if not csv_documents:
        logger.warning("CSV files were found but no documents were loaded.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    all_splits = text_splitter.split_documents(csv_documents)
    
    logger.info(f"Created {len(all_splits)} text chunks from CSV files")
    return all_splits

def load_text_files(FILE_DIRECTORY):
    loader = DirectoryLoader(FILE_DIRECTORY, glob="./*.txt", loader_cls=CustomTextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents were loaded. Check your 'new_articles' directory.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)

    if not all_splits:
        raise ValueError("No text chunks were created. The documents might be empty.")
    return all_splits

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

def load_text_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode the file {file_path} with any of the attempted encodings")

def safe_retrieve(query: str, retriever):
    docs = retriever.get_relevant_documents(query)
    if not docs:
        logger.warning(f"No relevant documents found for query: {query}")
        return "No relevant documents found for the query."
    logger.info(f"Retrieved {len(docs)} documents for query: {query}")
    for i, doc in enumerate(docs):
        logger.info(f"Document {i+1} content: {doc.page_content[:100]}...")  # Log first 100 chars of each doc
    return docs

def query_system(qa_chain, query: str):
    try:
        logger.info(f"Processing query: {query}")
        result = qa_chain({"query": query})
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
