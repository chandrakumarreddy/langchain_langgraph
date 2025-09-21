""""LLM"""

import os
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
        temperature=0,
        messages=[
            {"role": "user", "content": "write 3 line poem on wife"}
        ],
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(e)
