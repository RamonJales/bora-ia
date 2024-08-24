import os
import logging
import dotenv
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler
from bots.telegram.handlers import rag_answer, start


if __name__ == '__main__':
    dotenv.load_dotenv()
    BOT_TOKEN = os.environ['BOT_TELEGRAM_TOKEN']

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    application = ApplicationBuilder().token(BOT_TOKEN).build()

    rag_answer_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), rag_answer)
    start_handler = CommandHandler('start', start)

    application.add_handler(start_handler)
    application.add_handler(rag_answer_handler)

    application.run_polling()
