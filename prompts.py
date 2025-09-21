"""prompts"""

import os
from aiohttp.http_exceptions import HttpBadRequest
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


try:
    response = client.chat.completions.create(
        model="x-ai/grok-4-fast:free",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Capital of america"},
        ],
    )
    print(response.choices[0].message.content)
except HttpBadRequest as e:
    print(e)
except Exception as e:
    print(e)
