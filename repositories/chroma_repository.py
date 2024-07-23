"""
    ChromaRepository.py: handles all access to the chroma database
"""

__author__ = "Isaac Lourenço, Felipe Holanda"

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Iterable

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ChromaRepository:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR, embedding_model: str = CHROMA_EMBEDDING_MODEL):
        embedding_function = OpenAIEmbeddings(model=embedding_model)
        self._db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)


    def add_docs(self, docs: list[Document]):
        """
        add a list of langchain documents into the chromadb
        :param docs: list of langchain documents
        """
        if docs:
            self._db.add_documents(documents=docs)


    def remove_docs(self, file_paths: Iterable[str]):
        """
        hard remove all langchain documents at database based on its sources metadata
        :param file_paths: list of sources metadatas
        """
        if file_paths:
            for file_path in file_paths:
                docs = self._db.get(where={"file_path": file_path})
                self._db.delete(ids=docs["ids"])


    def as_retriever(self) -> VectorStoreRetriever:
        """
        :return: chromadb as a vector store retriever important to the RAG chain
        """
        return self._db.as_retriever()


    def get_file_paths(self) -> list[str]:
        """
        :return: list of all file_path metadata of all documents in the database
        """
        docs = self._db.get()
        sources = [metadata["file_path"] for metadata in docs["metadatas"]]

        return sources
