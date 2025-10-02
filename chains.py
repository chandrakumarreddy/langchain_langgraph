"""Sequential, paralle and conditional chains"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


client = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="x-ai/grok-4-fast:free",
)


prompt1 = PromptTemplate(
    template="""Generate details report on topic {topic}""", input_variables=["topic"])

prompt2 = PromptTemplate(
    template="""provide sumamry of the report in few lines\n {report}""", input_variables=["report"])


parser = StrOutputParser()

chain = prompt1 | client | parser | (
    lambda x: {"report": x}) | prompt2 | client | parser

try:
    print("Generating report...")
    FINAL_REPORT = ''
    for s in chain.stream({"topic": "AI"}):
        FINAL_REPORT += s
        print(s, end="")
    print("\n\nFinal report:\n", FINAL_REPORT)
except Exception as e:
    print(e)
