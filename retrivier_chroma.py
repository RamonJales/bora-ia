from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
retriever = db.as_retriever()