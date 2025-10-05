"""Runnable"""

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(openai_api_base="https://openrouter.ai/api/v1",
                 openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 model_name="openai/gpt-oss-20b:free")


parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

joke_chain = prompt1 | llm | parser


parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split()))
})

chain = joke_chain | parallel_chain

try:
    result = chain.invoke({'topic': 'AI'})
    print(result)
except Exception as e:
    print(e)
