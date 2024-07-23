"""
    chroma_service.py = implements rules to access the database
"""

__author__ = "Isaac Lourenço, Felipe Holanda"


import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import Iterable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from repositories.chroma_repository import ChromaRepository

load_dotenv()

KNOWLEDGE_PDF_DIR = os.getenv("KNOWLEDGE_PDF_DIR")


def load_pdfs(pdfs: Iterable[str]) -> list[Document]:
    """
    loads the pdfs into Documents
    :param pdfs: list of filepaths to pdfs files
    :return: list of Documents
    """
    if pdfs:
        docs = []
        for pdf_file in pdfs:
            loader = PyMuPDFLoader(pdf_file)
            docs.extend(loader.load())

        return docs


def split_docs(docs: list[Document]) -> list[Document]:
    """
    splits the documents into smaller sizes for the database
    :param docs: list of documents to split
    :return: list of split documents
    """
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits


class ChromaService:
    def __init__(self, chroma_repository: ChromaRepository = ChromaRepository(),
                 knowledge_directory: str = KNOWLEDGE_PDF_DIR):
        self._chroma_repository = chroma_repository
        self._knowledge_directory = knowledge_directory

    def _get_pdfs_paths_from_dir(self) -> list[str]:
        """
        gets a list of pdfs paths in the knowledge directory
        :return: list of file paths to the pdfs in the directory
        """

        files: list[str] = os.listdir(self._knowledge_directory)
        pdfs_paths: list[str] = [os.path.join(
                        self._knowledge_directory, pdf_file) for pdf_file in files if pdf_file.lower().endswith('.pdf')]

        return pdfs_paths

    def _compare_files(self) -> tuple[set[str], set[str]]:
        """
            gets mismatched files between the Chroma database and the directory and returns them as two sets

            :return: a tuple where the first member is the set of files only in the directory, and
            the second is the set of files only in the database
        """

        paths: set[str] = set(self._get_pdfs_paths_from_dir())

        sources: set[str] = set(self._chroma_repository.get_sources())
        
        directory_only = paths.difference(sources)
        database_only = sources.difference(paths)

        return directory_only, database_only


    def _update_knowledge(self):
        """
            add or remove RAG knowledge based on knowledge folder new or removed files
            if the file exists only at chromadb, it was removed
            if the file exists only at knowledge folder, it is a new one
        """
        directory_only, database_only = self._compare_files()

        self._chroma_repository.add_docs(split_docs(load_pdfs(directory_only)))
        self._chroma_repository.remove_docs(database_only)


    def load_retriever(self) -> VectorStoreRetriever:
        """
            :return: chromadb as a retriever to the RAG
        """
        self._update_knowledge()

        retriever = self._chroma_repository.as_retriever()

        return retriever

