# Custom RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for querying data (local) using large language models (local) and vector databases (local).

![Screenshot 2024-08-20 095115](https://github.com/user-attachments/assets/62ef5dbb-49ee-492b-8b58-304c0129246c)

TO RUN CODE FOR EACH DATA TYPE (eg. PDF or CSV or TEX) PLEASE FIRST DELETE THE "chroma_db" FOLDER AND RUN THE CODE.

## Features

- Load and process data from PDF, Text and CSV files
- Use advanced language models for query processing
- Efficient data retrieval using Chroma vector database
- User-friendly interface with Streamlit

## Installation

### 1. Create Python Environment

```
conda create -p venv python==3.10 -y
conda activate ./venv
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Download Models
Download the following models and place them in the models directory:

Gemma 2 2B: https://huggingface.co/NickTsaizer/gemma-2-2b-it-Q6_K-GGUF

Llama-2 7B Chat: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf

Meta Llama 3.1 8B Instruct: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Mistral 7B Instruct: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_S.gguf

Usage
1. Start the Backend Server

```
python main.py
```

2. Launch the Streamlit Frontend

```
streamlit run app.py
```

## Project Structure

main.py: FastAPI backend server
app.py: Streamlit frontend application
utils.py: Utility functions for data processing and RAG system
data/: Directory for storing CSV data files
models/: Directory for storing language models
chroma_db/: Directory for Chroma vector database

## Configuration

Adjust the FILE_DIRECTORY, MODEL_PATH, and CHROMA_PERSIST_DIRECTORY variables in main.py to match your setup.
Modify the chunk_size and chunk_overlap parameters in utils.py to optimize text splitting for your specific dataset.

## Troubleshooting

If you encounter issues with CSV loading, check the file encoding and ensure it's compatible with the system.
For performance issues, consider using a smaller language model or reducing the size of your dataset.
Check the logs for detailed error messages and debugging information.

## Contributing
Contributions to improve the project are welcome. Please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

## License


## Acknowledgments









