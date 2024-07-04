from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_docs():
    files = os.listdir('./data/ppcs')

    pdf_files = [pdf_file for pdf_file in files if pdf_file.lower().endswith('.pdf')]

    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join('./data/ppcs', pdf_file)
        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

    return docs

def initialize_chroma() -> None:

    docs = load_docs()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    Chroma.from_documents(documents=splits, persist_directory="./chroma_db", embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

def add_ppcs_pdf(*ppc_name : str) -> None:
    pdf_files = [pdf_file for pdf_file in ppc_name if pdf_file.lower().endswith('.pdf')]

    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join('./data/ppcs', pdf_file)
        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    db.add_documents(documents=docs)

add_ppcs_pdf("PPC_Engenharia_Mecatrnica_Bacharelado_-_Natal_-_Atualizado_em_2018-1.pdf")

