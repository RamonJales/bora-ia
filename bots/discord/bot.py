import os
import discord
import requests
from dotenv import load_dotenv

load_dotenv()
SIMBORA_API_URI = os.getenv('SIMBORA_API_URI')
BOT_DISCORD_TOKEN = os.getenv('BOT_DISCORD_TOKEN')
BOT_DISCORD_PREFIX = os.getenv('BOT_DISCORD_PREFIX')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith(BOT_DISCORD_PREFIX):
        query = message.content.removeprefix(BOT_DISCORD_PREFIX)
        response = requests.get(SIMBORA_API_URI, params={"query": query})
        await message.channel.send(response.json())

client.run(BOT_DISCORD_TOKEN)