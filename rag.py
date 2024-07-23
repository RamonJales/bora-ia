import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from persist_chroma import load_chroma
from services.chroma_service import ChromaService

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-8b-8192")

system_prompt = (
    "Você é um assistente para tarefas de perguntas e respostas."
    "Use os seguintes trechos de contexto recuperados para responder à pergunta."
    "Se você não souber a resposta, diga que não sabe."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)


def get_rag_chain():
    db = load_chroma()

    retriever = ChromaService().load_retriever()
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain