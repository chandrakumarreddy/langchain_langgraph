"""JSON formatting"""

import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


client = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="x-ai/grok-4-fast:free",
)


class Place(BaseModel):
    """place"""
    name: str = Field(description="The name of the place")
    location: str = Field(description="The location of the place")
    rating: float = Field(description="The rating of the place")
    price_range: str = Field(description="The price range of the place")
    description: str = Field(description="The description of the place")


parser = JsonOutputParser(pydantic_object=Place)

prompt_template = PromptTemplate(template="give me the details of the place {place}\n{format_instructions}", input_variables=[
                                 "place"], partial_variables={"format_instructions": parser.get_format_instructions()})

chain = prompt_template | client | parser

try:
    for s in chain.stream({"place": "The Eiffel Tower"}):
        print(s)
except Exception as e:
    print(e)
