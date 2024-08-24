import os
import requests
from telegram import Update
from telegram.ext import ContextTypes
from dotenv import load_dotenv


load_dotenv()
SIMBORA_API_URI = os.environ['SIMBORA_API_URI']


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start handler. Responde com uma mensagem inicial padrão.
    :param update: Objeto update representando os dados enviados pelo usuário
    :param context: Objeto de contexto do handler para ter acesso ao bot
    """
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Olá, eu sou o Simbora! Como posso te ajudar hoje?")


async def rag_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Envia uma mensagem para o usuário com a resposta da chamada ao RAG via API do Simbora.
    :param update: Objeto update representando os dados enviados pelo usuário
    :param context: Objeto de contexto do handler para ter acesso ao bot
    """
    response = requests.get(SIMBORA_API_URI, params={"query": update.message.text})
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response.json())