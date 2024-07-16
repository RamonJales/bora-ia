#!/usr/bin/env python

"""
    ChromaService.py = implements rules to access the database

"""

__author__ = "Isaac Lourenço, Felipe Holanda"


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from typing import Iterable

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ChromaRepository import ChromaRepository

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
        self.chroma_repository = chroma_repository
        self.knowledge_directory = knowledge_directory

    def get_pdfs_from_dir(self) -> list[str]:
        """
        gets a list of pdfs in the knowledge directory
        :return: list of file paths to the pdfs in the directory
        """

        files: list[str] = os.listdir(self.knowledge_directory)
        pdf_files: list[str] = [os.path.join(
                        self.knowledge_directory, pdf_file) for pdf_file in files if pdf_file.lower().endswith('.pdf')]

        return pdf_files

    def compare_files(self) -> tuple[set[str], set[str]]:
        """
            gets mismatched files between the Chroma database and the directory and returns them as two sets

            :return: a tuple where the first member is the set of files only in the directory, and
            the second is the set of files only in the database
        """

        pdf_files_set: set[str] = set(self.get_pdfs_from_dir())

        sources_set: set[str] = set([metadata["source"] for metadata in self.chroma_repository.db.get()["metadatas"]])

        directory_only = pdf_files_set.difference(sources_set)
        database_only = sources_set.difference(pdf_files_set)

        return directory_only, database_only

    def update_repository(self):
        new_files, removed_files = self.compare_files()

        self.chroma_repository.remove_docs(removed_files)

        self.chroma_repository.add_docs(split_docs(load_pdfs(new_files)))

    def load_retriever(self) -> VectorStoreRetriever:
        self.update_repository()

        retriever = self.chroma_repository.as_retriever()

        return retriever

