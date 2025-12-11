# 🧠 RAG Test Case Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green?style=for-the-badge&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=for-the-badge)

## 📝 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to automatically generate high-quality **Software Test Cases** from requirement documents (Word/Docx).

It leverages modern NLP techniques to parse requirements, chunk text intelligently, embed context using state-of-the-art models (Gemini, BGE-M3, Qwen), and retrieve relevant information to prompt a Large Language Model (LLM) for test case creation.

## ✨ Features

- **📄 Smart Document Parsing**: Extracts text and sections from `.docx` files using `python-docx`.
- **🧩 Advanced Chunking**: Custom hierarchical chunking strategy specialized for Requirement spec documents (Epics, Narratives, Acceptance Criteria).
- **🔎 Multi-Model Embeddings**: 
  - **Gemini**: Google's generative embeddings.
  - **BGE-M3**: Strong dense retrieval model.
  - **Qwen3**: Optimized multilingual embeddings.
- **💾 Vector Database**: Uses **ChromaDB** for efficient storage and retrieval of context.
- **🤖 LLM Integration**: Generates structured test cases (JSON) using **Ollama** (running local models like `gpt-oss:20b`).

## 📁 Project Structure

```bash
RAG_testcases/
├── 📄 rag.ipynb          # Main RAG pipeline, embedding benchmarks, and generation
├── 🐍 main.py            # Entry point for testing data loading and chunking
├── 🐍 load_data.py       # Module for reading Docx files
├── 🐍 chunking.py        # Custom logic for splitting requirement documents
├── 📝 prompt.txt         # System prompt for the LLM
├── 📂 chroma_db/         # Persisted Vector Database
└── 📄 requirements.txt   # Project dependencies
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    Recommendation: Use a virtual environment (e.g., `conda` or `venv`).

    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables**:
    Create a `.env` file in the root directory and add your keys (e.g., for Gemini):

    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

4.  **Install Ollama**:
    Ensure you have [Ollama](https://ollama.com/) installed and running. Pull the necessary models:

    ```bash
    ollama pull qwen3-embedding:0.6b
    ollama pull gpt-oss:20b
    # or other models specified in the notebook
    ```

## 🚀 Usage

### Running the Python Scripts
To test the data loading and chunking logic independently:

```bash
python main.py
```

### Running the RAG Pipeline
Open `rag.ipynb` in Jupyter Notebook or VS Code to interact with the full pipeline:
1.  **Load Data**: Parses the requirement document.
2.  **Generate Embeddings**: Runs and benchmarks different embedding models.
3.  **Retrieve & Generate**: Query the system (e.g., "Generate test cases for PSE1.4") and receive structured JSON output.

## 🤝 Contributing
Feel free to open issues or submit pull requests for improvements!

---
*Generated with ❤️ by Antigravity*
