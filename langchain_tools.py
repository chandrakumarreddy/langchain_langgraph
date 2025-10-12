"""Tools"""

# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# results = search_tool.invoke(
#     "Ind vs Aus womens world cup. Live Score of Austrilia in women's world cup today")
# print(results)

import os
from typing import Type
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_core.core_schema import arguments_schema

load_dotenv()


llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="openai/gpt-oss-20b:free",
)


@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of the two numbers
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Product of the two numbers
    """
    return a * b


# llm.bind_tools([add, multiply])
parser = StrOutputParser()

# chain = llm | parser

# print(chain.invoke("mutliply 29291 and 92? return only the output"))


# ---------------------------#
# using Structured tool
# ---------------------------#


class MultiplyInput(BaseModel):
    """Input for multiplication"""
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


@tool
def multiply_specs(specs: MultiplyInput) -> int:
    """
    Multiply two numbers

    Args:
        specs (MultiplyInput): Specifications for multiplication

    Returns:
        int: Product of the two numbers
    """
    return specs.a * specs.b


multiply_tool = StructuredTool.from_function(
    func=multiply_specs,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput)

llm.bind_tools([multiply_tool])

chain = llm | parser

# print(chain.invoke("mutliply 2 and 4? return only the output"))

# ---------------------------#
# using BaseTool
# ---------------------------#


class MultiplyToolBase(BaseTool):
    """
    Multiply two numbers
    """

    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool_base = MultiplyToolBase()

llm.bind_tools([multiply_tool_base])

chain = llm | parser

print(chain.invoke("mutliply 2 and 4? return only the output"))

# ---------------------------#
# Toolkit
# ---------------------------#


class MathKit:
    """
    MathKit is a toolkit for mathematical operations
    """

    def get_tools(self):
        """
        Get tools
        """
        return [add, multiply]


math_kit = MathKit()
math_kit_tools = math_kit.get_tools()

for tool in math_kit_tools:
    print(tool.name, tool.description)
