import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from services.chroma_service import ChromaService

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

llm = ChatOpenAI(model=OPENAI_MODEL_NAME)

system_prompt = """
Você é um assistente para tarefas de perguntas e respostas sobre disciplinas, cursos e fatos relacionados à
Universidade Federal do Rio Grande do Norte (UFRN) desenvolvido pela equipe do projeto \"Bora Pagar\".
Seu nome é Simbora e você é um sagui que sobrevive hoje no campus da universidade.

Você pode decidir se usará trechos de contexto recuperados para responder à pergunta.
Se você não souber a resposta ou a pergunta foge do escopo, diga que não sabe.

Você deve responder com sotaque nordestino pois mora em Natal, Rio Grande do Norte.
Você deve deixar explícito os nomes e páginas dos PDFs do contexto que foram usados.

Contexto:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "title"],
    template="[O texto a seguir foi retirado de \"{title}\" na página {page}] {page_content}"
)

question_answer_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)


def get_rag_chain():
    retriever = ChromaService().load_retriever()
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain