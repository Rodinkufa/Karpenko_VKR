import nest_asyncio
nest_asyncio.apply()

import asyncio
from datetime import datetime, timedelta, timezone
import pandas as pd
from telethon import TelegramClient

# Конфигурация
api_id = 24874969
api_hash = '7a055351ac0400bb75f42cee95399e52'
session_name = 'my_session'
channel_username = '@ifax_go'
moscow_tz = timezone(timedelta(hours=3))

def get_previous_working_day():
    today = datetime.now(moscow_tz).date()
    previous_day = today - timedelta(days=1)
    while previous_day.weekday() >= 5:
        previous_day -= timedelta(days=1)
    return previous_day

async def extract_news():
    client = TelegramClient(session_name, api_id, api_hash)
    await client.start()
    previous_day = get_previous_working_day()
    start = datetime.combine(previous_day, datetime.min.time(), tzinfo=moscow_tz)
    end = datetime.combine(previous_day, datetime.max.time(), tzinfo=moscow_tz)

    entity = await client.get_entity(channel_username)
    messages = []
    async for message in client.iter_messages(entity, limit=None):
        msg_time = message.date.astimezone(moscow_tz)
        if msg_time < start:
            break
        if start <= msg_time <= end:
            messages.append({'Date': msg_time.strftime('%Y-%m-%d %H:%M:%S'), 'Text': message.text})

    return pd.DataFrame(messages)
