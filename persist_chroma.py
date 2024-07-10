from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_docs():
    """TODO: O que faz essa função."""
    files = os.listdir('./data/ppcs')

    pdf_files = [pdf_file for pdf_file in files if pdf_file.lower().endswith('.pdf')]

    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join('./data/ppcs', pdf_file)
        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

    return docs

def initialize_chroma() -> Chroma:
    """TODO: O que faz essa função."""
    docs = load_docs()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    Chroma.from_documents(documents=splits, persist_directory="./chroma_db", embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

def get_docs_not_in_db(db : Chroma):
    """TODO: Documentar."""
    files = os.listdir('./data/ppcs')

    pdf_files = [pdf_file for pdf_file in files if pdf_file.lower().endswith('.pdf')]

    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join('./data/ppcs', pdf_file)

        docs_in_db = db.get(where={"source": file_path})

        if not docs_in_db['ids']:
            loader = PyMuPDFLoader(file_path)
            docs.extend(loader.load())

    return docs
def load_chroma() -> Chroma:
    """TODO: Documentar."""
    # FIXME: MODEL em .env
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))

    files = os.listdir('./data/ppcs')
    pdf_files = set([os.path.join('./data/ppcs', pdf_file) for pdf_file in files if pdf_file.lower().endswith('.pdf')])
    sources = set([metadata["source"] for metadata in db.get()["metadatas"]])

    not_changed_files = pdf_files.intersection(sources)
    new_ones = pdf_files.difference(not_changed_files)
    removed_ones = sources.difference(not_changed_files)

    for removed_one in removed_ones:
        docs = db.get(where={"source": removed_one})
        # print(docs)
        db.delete(ids=docs["ids"])

    docs = []
    for pdf_file in new_ones:
        loader = PyMuPDFLoader(pdf_file)
        docs.extend(loader.load())

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        db.add_documents(documents=splits)

    return db

