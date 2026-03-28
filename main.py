import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

if(os.environ.get("GOOGLE_API_KEY") is None):
    raise ValueError("GOOGLE_API_KEY is not set.")

if(os.environ.get("LANGCHAIN_API_KEY") is None):
    raise ValueError("LANGCHAIN_API_KEY is not set.")

client = Client()
print("Starting document loading...")

"""document loading"""
user_input = input("Enter a topic: ")
loader = WikipediaLoader(
    query=user_input,
    load_max_docs=2,
    lang="en"
)

docs = loader.load()
print(f"Documents loaded successfully: {len(docs)} documents")

"""text splitting"""
print("Creating text splits...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} text splits")


"""embedding"""
print("Creating embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

"""vector store"""
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="ai_wikipedia",
    persist_directory="./chroma_db"
)

print("Vector store created successfully")

"""retriever"""
retriever = vectorstore.as_retriever()
print("Retriever created successfully")

"""prompt template"""
print("Loading prompt template...")
prompt = client.pull_prompt("jclemens24/rag-prompt")
print("Prompt template loaded")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

"""RAG chain"""
print("Setting up Gemini model...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)
print("Language model created successfully")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain created successfully")

print("\nGenerating response...")
result = rag_chain.invoke(f"{user_input}")
print("\nResult:", result)