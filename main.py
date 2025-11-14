from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# Loading Text
loader = TextLoader("speech.txt")
document = loader.load()
print("LOADED TEXT.")


# Chunking
text_splitter = CharacterTextSplitter(
    separator=". ",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(document)
print("TEXT CHUNKING COMPLETE.")


# Embedder
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("EMBEDDER LOADED.")


# Embedding and Storing in Vector DB
db = Chroma.from_documents(chunks, embeddings, persist_directory="chrome_db")
print("EMBEDDING DONE & STORED IN CHROMA DB.")


# Retriever
retriever =  db.as_retriever(search_kwargs = {"k":3})
print("RETRIEVER LOADED.")


# LLM
llm = OllamaLLM(model="mistral")
print("LLM LOADED.")
# LLM TESTING
# print(llm.invoke("Hello, what are you?"))


"""
Deprecated RAG: `RetrievalQA` from `langchain_classic` is deprecated.

LangChain’s modern RAG approach uses:
- `dynamic_prompt` middleware
- `ModelRequest`
- `create_agent(...)`

Updated docs: https://docs.langchain.com/oss/python/langchain/rag#rag-chains

This script continues using the classic/legacy RAG chain for now.

"""


# RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# QA Loop
while True:
    query = input("\nAsk a question or (exit/quit): ")
    if query.lower() in ["exit","quit"]:
        break
    result = qa.invoke({"query": query})
    print("\nAnswer:",result["result"])