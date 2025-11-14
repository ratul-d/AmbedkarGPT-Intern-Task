# **AmbedkarGPT Q&A System**

## **Overview**

This repository contains a fully local Retrieval-Augmented Generation (RAG) pipeline built using LangChain, Sentence-Transformers embeddings, Chroma vector storage, and an Ollama-hosted LLM. The system ingests a speech by Dr. B. R. Ambedkar, stores semantic representations locally, and enables a command-line interface for answering questions strictly grounded in the provided document. The entire workflow runs offline without relying on any external APIs or cloud services.

The project demonstrates the full lifecycle of a text-based Q&A system:

1. **Document ingestion**
2. **Chunking and preprocessing**
3. **Embedding generation using a Sentence-Transformers (SBERT) model**
4. **Vector database storage (Chroma)**
5. **Retriever configuration**
6. **LLM-powered RAG query answering**

This implementation uses legacy `RetrievalQA` from `langchain_classic` for instructional purposes, although LangChain’s modern middleware-based RAG architecture is recommended for production.

---

## **Project Structure**

```
.
├── speech.txt               # Source document used for Q&A
├── main.py                  # Primary RAG implementation
├── requirements.txt         # Requirements
└── README.md                # Technical documentation
```

---

## **Prerequisites**

### **1. Python Environment**

* Python 3.8 or higher is recommended.

### **2. Required Python Packages**

Install all dependencies:

```bash
pip install -r requirements.txt
```

### **3. Ollama Installation**

Install Ollama from:
[https://ollama.com](https://ollama.com)

Verify installation:

```bash
ollama --version
```

### **4. Download the Required Model**

This project uses the *Mistral* model:

```bash
ollama pull mistral
```

Ensure the model is successfully downloaded before running the script.

---

## **Pipeline Architecture**

### **1. Document Loading**

The `TextLoader` ingests the `speech.txt` file and constructs a LangChain document.

```python
loader = TextLoader("speech.txt")
document = loader.load()
```


### **2. Text Chunking**

`CharacterTextSplitter` divides the document into semantically manageable segments.

Configuration:

* `chunk_size`: 250 characters
* `chunk_overlap`: 50 characters
* `separator`: sentence-level split (`". "`)

This balances retrieval accuracy and memory efficiency.


### **3. Embedding Generation**

Embeddings are computed using:

```
sentence-transformers/all-MiniLM-L6-v2
```

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```


### **4. Vector Storage**

ChromaDB is used as the persistent vector store.

```python
db = Chroma.from_documents(chunks, embeddings, persist_directory="chrome_db")
```


### **5. Retrieval Mechanism**

A similarity-based retriever is created for top-k retrieval.

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
```


### **6. LLM Backend (Ollama)**

Inference is powered by an Ollama-hosted instance of the Mistral model.

```python
llm = OllamaLLM(model="mistral")
```

### **7. RetrievalQA Chain**

The `RetrievalQA` chain is utilized here as the core mechanism for generating answers from retrieved context.


```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)
```


### **8. CLI Question-Answer Loop**

The script continuously accepts user queries until termination.

```python
while True:
    query = input("\nAsk a question or (exit/quit): ")
    if query.lower() in ["exit","quit"]:
        break
    result = qa.invoke({"query": query})
    print("\nAnswer:",result["result"])
```

---

## **Running the System**

### **Step 1: Place `speech.txt` in the project directory**

Ensure the file is named `speech.txt` and located in the same directory as the Python script.
This file serves as the sole knowledge source for the Q&A system.

### **Step 2: Start the Ollama server**

Before running the main script, the local LLM service must be active.
In a separate terminal, start the Ollama server:

```bash
ollama serve
```

This keeps the model available for inference during your RAG pipeline execution.


### **Step 3: Run the Python script**

With Ollama running, execute the program:

```bash
python main.py
```

On startup, the script will:

* Load the speech text
* Split it into chunks
* Generate Sentence-Transformer embeddings
* Build or reuse the Chroma vector database
* Connect to the locally running Ollama model


### **Step 4: Interact with the CLI**

After initialization, you can begin asking questions:

```
Ask a question or (exit/quit): What did Ambedkar say about democracy?
Answer: ...
```

You may ask any question related to the content of `speech.txt`.
Enter `exit` or `quit` to end the session.

---

## **Limitations**

* Context limited strictly to `speech.txt`.
* Inference speed depends on local hardware and Ollama configuration.
* Chunking strategy is basic and may require refinement for larger datasets.

---

## **License**

This project is released for educational and research purposes.
Please ensure compliance with license terms for the following components:

* LangChain
* Sentence Transformers
* ChromaDB
* Ollama Models

---