# RAG with Web Search 

![Screenshot (20)](https://github.com/user-attachments/assets/70bb6cf8-dbaf-4c53-abfd-dedeafeaa26b)
![Screenshot (23)](https://github.com/user-attachments/assets/b7327697-d5a4-4089-aac6-6188d4e09335)

---

## 🚀 Setup Instructions

Follow these steps to set up and run this project for the first time:

### 1. Clone the repository and set up a virtual environment (optional but recommended)
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies
```sh
pip install -r requirements.txt
```

### 3. Install and Run Ollama
- Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
- Start the Ollama server:
  ```sh
  ollama serve
  ```
  (Or just `ollama` if it starts the server by default)

### 4. Pull Required Ollama Models
- Pull at least one LLM model (e.g. llama3, mistral, phi3):
  ```sh
  ollama pull llama3
  ollama pull mistral
  ollama pull phi3
  ```
- Pull the embedding model required by the app:
  ```sh
  ollama pull nomic-embed-text
  ```

### 5. Run the Streamlit App
```sh
streamlit run src/main_app.py
```