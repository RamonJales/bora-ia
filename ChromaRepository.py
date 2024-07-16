#!/usr/bin/env python

"""
    ChromaRepository.py: handles all access to the chroma database

"""

__author__ = "Isaac Lourenço, Felipe Holanda"

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import Iterable

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ChromaRepository:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR, embedding_model: str = CHROMA_EMBEDDING_MODEL):
        embedding_function = OpenAIEmbeddings(model=embedding_model)
        self.db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    def add_docs(self, docs: list[Document]):
        if docs:
            self.db.add_documents(documents=docs)

    def remove_docs(self, sources: Iterable[str]):
        if sources:
            for source in sources:
                docs = self.db.get(where={"source": source})
                self.db.delete(ids=docs["ids"])

    def as_retriever(self) -> VectorStoreRetriever:
        return self.db.as_retriever()

    def get_sources(self) -> list[str]:
        docs = self.db.get()
        sources = [metadata["source"] for metadata in docs["metadatas"]]

        return sources
