from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import typer

# Loading Text
loader = TextLoader("speech.txt")
document = loader.load()
typer.echo(typer.style("LOADED TEXT.",fg=typer.colors.BRIGHT_GREEN))


# Chunking
text_splitter = CharacterTextSplitter(
    separator=". ",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(document)
typer.echo(typer.style("TEXT CHUNKING COMPLETE.",fg=typer.colors.BRIGHT_GREEN))


# Embedder
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
typer.echo(typer.style("EMBEDDER LOADED.",fg=typer.colors.GREEN))


# Embedding and Storing in Vector DB
db = Chroma.from_documents(chunks, embeddings, persist_directory="chrome_db")
typer.echo(typer.style("EMBEDDING DONE & STORED IN CHROMA DB.",fg=typer.colors.BRIGHT_GREEN))


# Retriever
retriever =  db.as_retriever(search_kwargs = {"k":3})
typer.echo(typer.style("RETRIEVER LOADED.",fg=typer.colors.BRIGHT_GREEN))


# LLM
llm = OllamaLLM(model="mistral")
typer.echo(typer.style("LLM LOADED.",fg=typer.colors.BRIGHT_GREEN))
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
    typer.echo(typer.style("\nAsk a question or (exit/quit): ", fg=typer.colors.BRIGHT_GREEN), nl=False)
    query = input()
    if query.lower() in ["exit","quit"]:
        break

    retrieved_docs = retriever.invoke(query)
    typer.echo(typer.style("\nRetrieved Chunks: ",fg=typer.colors.BRIGHT_GREEN))
    for i, doc in enumerate(retrieved_docs,start=1):
        print(f"Chunk {i}:", doc.page_content)

    result = qa.invoke({"query": query})
    typer.echo(typer.style("\nResponse: ", fg=typer.colors.BRIGHT_GREEN))
    print(result["result"])