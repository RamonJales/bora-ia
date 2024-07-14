from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_PDF_DIR = os.getenv("KNOWLEDGE_PDF_DIR")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL")


def load_docs() -> list[Document]:
    """
    loads the pdfs in the files directory into a list

    Return: list[Document]
    """

    files: list[str] = os.listdir(KNOWLEDGE_PDF_DIR)

    pdf_files: list[str] = [pdf_file for pdf_file in files if pdf_file.lower().endswith('.pdf')]

    docs: list[Document] = []
    for pdf_file in pdf_files:
        file_path: str = os.path.join(KNOWLEDGE_PDF_DIR, pdf_file)
        loader = PyMuPDFLoader(file_path)
        docs.extend(loader.load())

    return docs


def get_changed_files(db: Chroma, dir: str) -> tuple[set[str], set[str]]:
    """
        gets mismatched files between the Chroma database and the directory and returns them as two sets

        db: a Chroma database
        dir: the directory to compare files to

        Returns: a tuple where the first member is the set of files in dir but not in db, and 
        the second is the set of files in db but not in dir
    
    """

    files: list[str] = os.listdir(dir)
    pdf_files: set[str] = set([os.path.join(dir, pdf_file) for pdf_file in files if pdf_file.lower().endswith('.pdf')])
    sources = set([metadata["source"] for metadata in db.get()["metadatas"]])

    not_changed_files = pdf_files.intersection(sources)
    new_ones = pdf_files.difference(not_changed_files)
    removed_ones = sources.difference(not_changed_files)

    return new_ones, removed_ones


def remove_docs(db: Chroma, removed_ones: list[str]) -> None:
    """
        Removes the documents listed from the database

        db: a Chroma database
        removed_ones: list of documents to remove
    """
    for removed_one in removed_ones:
        docs = db.get(where={"source": removed_one})
        # print(docs)
        db.delete(ids=docs["ids"])


def add_docs(db: Chroma, new_ones: list[str]) -> None:
    """
        adds the documents listed as embeddings to the chroma database

        db: a Chroma database
        removed_ones: list of documents to add
    """

    docs = []
    for pdf_file in new_ones:
        loader = PyMuPDFLoader(pdf_file)
        docs.extend(loader.load())

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        db.add_documents(documents=splits)


def load_chroma() -> Chroma:
    """
        load chroma database from persist directory and updates based on files directory

        Returns: a Chroma database
    """

    db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=OpenAIEmbeddings(model=CHROMA_EMBEDDING_MODEL))

    new_ones, removed_ones = get_changed_files(db, KNOWLEDGE_PDF_DIR)

    remove_docs(db, removed_ones)
    add_docs(db, new_ones)

    return db
