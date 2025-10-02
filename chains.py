"""Sequential, paralle and conditional chains"""

import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel
from pydantic import BaseModel, Field

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


parallel_1 = PromptTemplate(
    template="""generate simple and short notes from the text \n {text}""", input_variables=["text"])

parallel_2 = PromptTemplate(
    template="""Generate 4 question and answers from the provided text \n {text}""", input_variables=["text"])

parallel_3 = PromptTemplate(
    template="""Merge provided notes and qn into single document \n notes->{notes} and quiz->{quiz}""", input_variables=["text"])

parallel_chain_output = RunnableParallel({
    "notes": parallel_1 | client | parser,
    "quiz": parallel_2 | client | parser
})

merge_chain = parallel_3 | client | parser

parellel_chain = parallel_chain_output | merge_chain

TEXT = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

try:
    print("Generating notes and quiz...")
    for s in parellel_chain.stream({"text": TEXT}):
        print(s, end='')
except Exception as e:
    print(e)


class Sentiment(BaseModel):
    """sentiment"""
    sentiment: Literal["Positive", "Negative"] = Field(
        description="Provide sentiment of the text")


sentiment_parser = PydanticOutputParser(pydantic_object=Sentiment)

sentiment_template = PromptTemplate(
    template="""Classify the sentiment of the following feedback text into postive or negative\n {feedback}\n {format_instructions}""",
    input_variables=["feedback"],
    partial_variables={"format_instructions": sentiment_parser.get_format_instructions()})

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

sentiment_runnable = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | client | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | client | parser),
    RunnableLambda(lambda x: "Sentiment not found")
)

sentiment_chain = sentiment_template | client | sentiment_parser | sentiment_runnable

try:
    print("Generating sentiment...")
    for s in sentiment_chain.stream({"feedback": "The product is great!"}):
        print(s, end='')
except Exception as e:
    print(e)
